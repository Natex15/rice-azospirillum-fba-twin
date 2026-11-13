"""model_builder.py: Rice-Azospirillum Two-Compartment Metabolic Model Architecture.

Couples an Indica rice host (Oryza sativa indica) with a rhizosphere diazotrophic
bacterium (Azospirillum brasilense) across 4 biological compartments with 24 reactions.
Features 100% standalone SciPy HiGHS fallback LP solver, Two-Stage Slack Carbon
Starvation Detection, full COBRApy compatibility, and pure-Python SBML/JSON serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog

try:
  import cobra
  from cobra import Metabolite, Model, Reaction

  HAS_COBRA = True
except ImportError:
  HAS_COBRA = False


# ==============================================================================
# FBA Solution Data Contract
# ==============================================================================


@dataclass
class FBASolution:
  """Standard FBA solution contract returned by TwoCompartmentMetabolicModel.optimize().

  Conforms strictly to PROJECT.md and SCOPE.md interface contracts.

  Attributes:
      status: Solver status string: "optimal", "infeasible", or "starved".
      objective_value: Numerical value of evaluated objective (e.g. BIOMASS_rice
        flux). Set to 0.0 under carbon starvation or infeasibility.
      fluxes: Dictionary mapping reaction ID string to solved flux value (mmol
        gDW^-1 h^-1).
      is_starved: Boolean flag set to True when host cellular maintenance
        (ATPM_r >= 2.5) cannot be satisfied by available photosynthetic carbon.
      atpm_deficit: Shortfall in ATP maintenance (mmol gDW^-1 h^-1): max(0.0,
        2.5 - v_atp_actual).
  """

  status: str
  objective_value: float
  fluxes: dict[str, float]
  is_starved: bool
  atpm_deficit: float

  # Convenience properties for downstream consumers (simulation_engine, tests, visualizer)
  @property
  def is_feasible(self) -> bool:
    return self.status == "optimal" and not self.is_starved

  @property
  def starvation_deficit(self) -> float:
    return self.atpm_deficit

  @property
  def biomass_rice(self) -> float:
    return self.fluxes.get("BIOMASS_rice", 0.0)

  @property
  def biomass_ab(self) -> float:
    return self.fluxes.get("BIOMASS_ab", 0.0)

  @property
  def suc_syn_r(self) -> float:
    return self.fluxes.get("SUC_SYN_r", 0.0)

  @property
  def sweet11_export(self) -> float:
    return self.fluxes.get("SWEET11_export", 0.0)

  @property
  def nitrogenase_ab(self) -> float:
    return self.fluxes.get("NITROGENASE_ab", 0.0)

  @property
  def nh4_excretion_ab(self) -> float:
    return self.fluxes.get("NH4_excretion_ab", 0.0)

  @property
  def nh4_uptake_rice(self) -> float:
    return self.fluxes.get("NH4_uptake_rice", 0.0)

  @property
  def suc_uptake_ab(self) -> float:
    return self.fluxes.get("SUC_uptake_ab", 0.0)

  @property
  def atpm_r(self) -> float:
    return self.fluxes.get("ATPM_r", 0.0)

  @property
  def atpm_ab(self) -> float:
    return self.fluxes.get("ATPM_ab", 0.0)

  @property
  def ex_photon_r(self) -> float:
    return self.fluxes.get("EX_photon_r", 0.0)

  @property
  def ex_nh4_soil(self) -> float:
    return self.fluxes.get("EX_nh4_soil", 0.0)


# ==============================================================================
# Model Entities & Dict Containers
# ==============================================================================


@dataclass
class MetaboliteData:
  """Domain representation of a chemical metabolite pool."""

  id: str
  name: str
  compartment: str
  formula: str = ""
  charge: int = 0


class ReactionData:
  """Domain representation of a biochemical or boundary reaction."""

  def __init__(
      self,
      id: str,
      name: str,
      stoichiometry: dict[str, float],
      lower_bound: float = 0.0,
      upper_bound: float = 1000.0,
      objective_coefficient: float = 0.0,
  ):
    self.id = id
    self.name = name
    self.stoichiometry = dict(stoichiometry)
    self.lower_bound = float(lower_bound)
    self.upper_bound = float(upper_bound)
    self.objective_coefficient = float(objective_coefficient)

  @property
  def metabolites(self) -> dict[str, float]:
    return dict(self.stoichiometry)

  def __repr__(self) -> str:
    return (
        f"<ReactionData {self.id}: bounds=({self.lower_bound},"
        f" {self.upper_bound})>"
    )


class ReactionDict(dict):
  """Dictionary of reactions supporting COBRApy-style get_by_id access."""

  def get_by_id(self, rxn_id: str) -> ReactionData:
    if rxn_id in self:
      return self[rxn_id]
    raise KeyError(f"Reaction '{rxn_id}' not found in model.")


class MetaboliteDict(dict):
  """Dictionary of metabolites supporting COBRApy-style get_by_id access."""

  def get_by_id(self, met_id: str) -> MetaboliteData:
    if met_id in self:
      return self[met_id]
    raise KeyError(f"Metabolite '{met_id}' not found in model.")


# ==============================================================================
# Core Metabolic Model Class
# ==============================================================================


class TwoCompartmentMetabolicModel:
  """Encapsulates reactions, metabolites, S matrix, bounds, and solver interfaces

  for the Rice–Azospirillum two-compartment metabolic system.
  """

  def __init__(
      self,
      model_id: str = "Rice_Azospirillum_Symbiosis",
      name: str = "Rice Azospirillum Two-Compartment Model",
  ):
    self.model_id = model_id
    self.name = name
    self.compartments: dict[str, str] = {
        "r_shoot": "Rice Shoot Canopy",
        "r_root": "Rice Root System",
        "rhizo": "Rhizosphere Shared Exchange Pool",
        "a_bras": "Azospirillum brasilense Symbiont",
    }
    self.metabolites = MetaboliteDict()
    self.reactions = ReactionDict()
    self.objective_reaction: str = "BIOMASS_rice"
    self.objective_direction: str = "max"

  def add_metabolite(
      self,
      met_id: str,
      name: str = "",
      compartment: str = "",
      formula: str = "",
      charge: int = 0,
  ) -> None:
    self.metabolites[met_id] = MetaboliteData(
        id=met_id,
        name=name or met_id,
        compartment=compartment,
        formula=formula,
        charge=charge,
    )

  def add_reaction(
      self,
      rxn_id: str,
      name: str = "",
      stoichiometry: dict[str, float] | None = None,
      lower_bound: float = 0.0,
      upper_bound: float = 1000.0,
      objective_coefficient: float = 0.0,
  ) -> None:
    stoich = dict(stoichiometry or {})
    self.reactions[rxn_id] = ReactionData(
        id=rxn_id,
        name=name or rxn_id,
        stoichiometry=stoich,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        objective_coefficient=float(objective_coefficient),
    )
    if objective_coefficient != 0.0:
      self.objective_reaction = rxn_id

  @property
  def reaction_ids(self) -> list[str]:
    return list(self.reactions.keys())

  @property
  def metabolite_ids(self) -> list[str]:
    return list(self.metabolites.keys())

  @property
  def N(self) -> int:
    return len(self.reactions)

  @property
  def M(self) -> int:
    return len(self.metabolites)

  @property
  def lb(self) -> np.ndarray:
    return np.array(
        [r.lower_bound for r in self.reactions.values()], dtype=np.float64
    )

  @lb.setter
  def lb(self, values: Any) -> None:
    arr = np.asarray(values, dtype=np.float64)
    for j, r in enumerate(self.reactions.values()):
      r.lower_bound = float(arr[j])

  @property
  def ub(self) -> np.ndarray:
    return np.array(
        [r.upper_bound for r in self.reactions.values()], dtype=np.float64
    )

  @ub.setter
  def ub(self, values: Any) -> None:
    arr = np.asarray(values, dtype=np.float64)
    for j, r in enumerate(self.reactions.values()):
      r.upper_bound = float(arr[j])

  @property
  def S(self) -> np.ndarray:
    m = len(self.metabolites)
    n = len(self.reactions)
    mat = np.zeros((m, n), dtype=np.float64)
    met_map = {m_id: i for i, m_id in enumerate(self.metabolites.keys())}
    for j, rxn in enumerate(self.reactions.values()):
      for m_id, coeff in rxn.stoichiometry.items():
        if m_id in met_map:
          mat[met_map[m_id], j] = coeff
    return mat

  def get_reaction_bounds(self, reaction_id: str) -> tuple[float, float]:
    if reaction_id not in self.reactions:
      raise KeyError(f"Reaction '{reaction_id}' not found in model.")
    r = self.reactions[reaction_id]
    return (float(r.lower_bound), float(r.upper_bound))

  def set_reaction_bounds(self, reaction_id: str, lb: float, ub: float) -> None:
    if reaction_id not in self.reactions:
      raise KeyError(f"Reaction '{reaction_id}' not found in model.")
    if lb > ub:
      raise ValueError(
          f"Lower bound ({lb}) cannot exceed upper bound ({ub}) for reaction"
          f" '{reaction_id}'."
      )
    r = self.reactions[reaction_id]
    r.lower_bound = float(lb)
    r.upper_bound = float(ub)

  def set_genotype(
      self,
      genotype: str,
      lb_custom: float | None = None,
      ub_custom: float | None = None,
  ) -> None:
    """Configures SWEET11_export bounds corresponding to OsSWEET11 genetic profiles."""
    norm = genotype.strip().lower().replace("_", "").replace(" ", "")
    if norm in ("wt", "ossweet11wt"):
      self.set_reaction_bounds("SWEET11_export", 0.0, 10.0)
      if "ATPM_ab" in self.reactions:
        self.set_reaction_bounds("ATPM_ab", 0.5, 1000.0)
    elif norm in ("oe", "ossweet11oe"):
      lb = 2.0 if lb_custom is None else float(lb_custom)
      ub = 20.0 if ub_custom is None else float(ub_custom)
      self.set_reaction_bounds("SWEET11_export", lb, ub)
      if "ATPM_ab" in self.reactions:
        self.set_reaction_bounds("ATPM_ab", 0.5, 1000.0)
    elif norm in ("knockdown", "ossweet11knockdown", "kd", "ossweet11kd"):
      self.set_reaction_bounds("SWEET11_export", 0.0, 3.0)
      if "ATPM_ab" in self.reactions:
        self.set_reaction_bounds("ATPM_ab", 0.5, 1000.0)
    elif norm in ("ko", "ossweet11ko"):
      self.set_reaction_bounds("SWEET11_export", 0.0, 0.0)
      if "ATPM_ab" in self.reactions:
        # KO eliminates sucrose exudation; relax bacterial maintenance to prevent false host failure
        self.set_reaction_bounds("ATPM_ab", 0.0, 1000.0)
    elif norm == "custom":
      if lb_custom is None or ub_custom is None:
        raise ValueError(
            "lb_custom and ub_custom must be specified for 'custom' genotype."
        )
      self.set_reaction_bounds(
          "SWEET11_export", float(lb_custom), float(ub_custom)
      )
    else:
      raise ValueError(f"Unrecognized genotype profile: '{genotype}'")

    if lb_custom is not None and norm != "custom":
      _, ub = self.get_reaction_bounds("SWEET11_export")
      self.set_reaction_bounds("SWEET11_export", float(lb_custom), ub)
    if ub_custom is not None and norm != "custom":
      lb, _ = self.get_reaction_bounds("SWEET11_export")
      self.set_reaction_bounds("SWEET11_export", lb, float(ub_custom))

  def set_light_intensity(self, pct: float) -> None:
    """Scales EX_photon_r upper bound where 100% = 100.0 mmol gDW^-1 h^-1."""
    val = max(0.0, float(pct))
    self.set_reaction_bounds("EX_photon_r", 0.0, val)

  def set_soil_ammonium(self, level: float) -> None:
    """Sets EX_nh4_soil upper bound in mmol gDW^-1 h^-1."""
    val = max(0.0, float(level))
    self.set_reaction_bounds("EX_nh4_soil", 0.0, val)

  def to_scipy_lp(
      self,
  ) -> tuple[
      np.ndarray,
      np.ndarray,
      np.ndarray,
      list[tuple[float, float]],
      list[str],
      list[str],
  ]:
    """Exports canonical Linear Programming matrices for SciPy HiGHS solver.

    Returns:
        tuple: (c, S, b_eq, bounds, rxn_ids, met_ids)
    """
    rxn_ids = list(self.reactions.keys())
    met_ids = list(self.metabolites.keys())
    m = len(met_ids)
    n = len(rxn_ids)
    met_map = {m_id: i for i, m_id in enumerate(met_ids)}

    S = np.zeros((m, n), dtype=np.float64)
    bounds = []
    c = np.zeros(n, dtype=np.float64)

    for j, r_id in enumerate(rxn_ids):
      rxn = self.reactions[r_id]
      bounds.append((float(rxn.lower_bound), float(rxn.upper_bound)))
      if r_id == self.objective_reaction:
        c[j] = -1.0 if self.objective_direction == "max" else 1.0
      for met_id, stoich in rxn.stoichiometry.items():
        if met_id in met_map:
          S[met_map[met_id], j] = stoich

    b_eq = np.zeros(m, dtype=np.float64)
    return (c, S, b_eq, bounds, rxn_ids, met_ids)

  def optimize(
      self,
      objective: str = "BIOMASS_rice",
      maximize: bool = True,
      solver: str = "highs",
  ) -> FBASolution:
    """Solves steady-state FBA with automatic SciPy HiGHS fallback and

    Two-Stage Slack Carbon Starvation Detection.
    """
    self.objective_reaction = objective
    self.objective_direction = "max" if maximize else "min"
    c, S, b_eq, bounds, rxn_ids, met_ids = self.to_scipy_lp()

    if objective not in rxn_ids:
      raise KeyError(f"Objective reaction '{objective}' not found in model.")
    obj_idx = rxn_ids.index(objective)

    # Stage 1: Standard Primal LP
    res1 = linprog(
        c=c,
        A_eq=S,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "disp": False,
            "presolve": True,
            "primal_feasibility_tolerance": 1e-7,
            "dual_feasibility_tolerance": 1e-7,
        },
    )

    if res1.status == 0:
      obj_val = float(res1.x[obj_idx]) if maximize else -float(res1.x[obj_idx])
      fluxes = {r_id: float(res1.x[j]) for j, r_id in enumerate(rxn_ids)}
      return FBASolution(
          status="optimal",
          objective_value=obj_val,
          fluxes=fluxes,
          is_starved=False,
          atpm_deficit=0.0,
      )

    # Stage 2: Two-Stage Slack Carbon Starvation Detection
    atpm_idx = rxn_ids.index("ATPM_r") if "ATPM_r" in rxn_ids else None
    if atpm_idx is not None:
      bounds_relaxed = list(bounds)
      _, orig_ub = bounds_relaxed[atpm_idx]
      bounds_relaxed[atpm_idx] = (0.0, orig_ub)
      # Also relax bacterial maintenance so bacterial carbon deficit does not mask host starvation
      if "ATPM_ab" in rxn_ids:
        ab_idx = rxn_ids.index("ATPM_ab")
        _, ab_ub = bounds_relaxed[ab_idx]
        bounds_relaxed[ab_idx] = (0.0, ab_ub)

      c_starve = np.zeros(len(rxn_ids), dtype=np.float64)
      c_starve[atpm_idx] = -1.0  # Maximize ATPM_r

      res2 = linprog(
          c=c_starve,
          A_eq=S,
          b_eq=b_eq,
          bounds=bounds_relaxed,
          method="highs",
          options={
              "disp": False,
              "presolve": True,
              "primal_feasibility_tolerance": 1e-7,
              "dual_feasibility_tolerance": 1e-7,
          },
      )
      if res2.status == 0:
        v_atp_actual = float(res2.x[atpm_idx])
        if v_atp_actual < 2.5 - 1e-5:
          deficit = max(0.0, 2.5 - v_atp_actual)
          fluxes = {r_id: float(res2.x[j]) for j, r_id in enumerate(rxn_ids)}
          fluxes[objective] = 0.0
          return FBASolution(
              status="starved",
              objective_value=0.0,
              fluxes=fluxes,
              is_starved=True,
              atpm_deficit=deficit,
          )
        else:
          # Maintenance can be satisfied; external conflict caused infeasibility
          return FBASolution(
              status="infeasible",
              objective_value=0.0,
              fluxes={r_id: 0.0 for r_id in rxn_ids},
              is_starved=False,
              atpm_deficit=0.0,
          )
      else:
        # If even relaxed LP is infeasible (e.g. forced SWEET11 export lb exceeds total produced carbon),
        # host experiences lethal carbon starvation.
        return FBASolution(
            status="starved",
            objective_value=0.0,
            fluxes={r_id: 0.0 for r_id in rxn_ids},
            is_starved=True,
            atpm_deficit=2.5,
        )

    # Structural unresolvable infeasibility
    return FBASolution(
        status="infeasible",
        objective_value=0.0,
        fluxes={r_id: 0.0 for r_id in rxn_ids},
        is_starved=True,
        atpm_deficit=2.5,
    )

  def to_cobra(self) -> Any:
    """Builds and returns a native COBRApy Model instance."""
    if not HAS_COBRA:
      raise ImportError(
          "COBRApy is not installed in the current Python environment."
      )
    cobra_model = Model(self.model_id, name=self.name)
    cobra_model.compartments = dict(self.compartments)

    cobra_mets = {}
    for m_data in self.metabolites.values():
      cm = Metabolite(
          id=m_data.id,
          name=m_data.name,
          compartment=m_data.compartment,
          formula=m_data.formula,
          charge=m_data.charge,
      )
      cobra_mets[m_data.id] = cm
    cobra_model.add_metabolites(list(cobra_mets.values()))

    cobra_rxns = []
    for r_data in self.reactions.values():
      cr = Reaction(
          id=r_data.id,
          name=r_data.name,
          lower_bound=r_data.lower_bound,
          upper_bound=r_data.upper_bound,
      )
      cr_stoich = {
          cobra_mets[m_id]: coeff
          for m_id, coeff in r_data.stoichiometry.items()
          if m_id in cobra_mets
      }
      cr.add_metabolites(cr_stoich)
      cobra_rxns.append(cr)
    cobra_model.add_reactions(cobra_rxns)
    if self.objective_reaction in cobra_model.reactions:
      cobra_model.objective = self.objective_reaction
    return cobra_model

  def to_json(self, filepath: str) -> None:
    """Serializes model to standard COBRA JSON format."""
    data = {
        "id": self.model_id,
        "name": self.name,
        "compartments": dict(self.compartments),
        "metabolites": [
            {
                "id": m.id,
                "name": m.name,
                "compartment": m.compartment,
                "formula": m.formula,
                "charge": m.charge,
            }
            for m in self.metabolites.values()
        ],
        "reactions": [
            {
                "id": r.id,
                "name": r.name,
                "metabolites": dict(r.stoichiometry),
                "lower_bound": float(r.lower_bound),
                "upper_bound": float(r.upper_bound),
                "objective_coefficient": float(
                    1.0 if r.id == self.objective_reaction else 0.0
                ),
            }
            for r in self.reactions.values()
        ],
        "genes": [],
        "lb": [float(r.lower_bound) for r in self.reactions.values()],
        "ub": [float(r.upper_bound) for r in self.reactions.values()],
        "S": self.S.tolist(),
        "reaction_ids": list(self.reactions.keys()),
        "metabolite_ids": list(self.metabolites.keys()),
    }
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)

  @classmethod
  def from_json(cls, filepath_or_buffer: Any) -> TwoCompartmentMetabolicModel:
    """Deserializes model from JSON file path, string buffer, or dictionary."""
    if isinstance(filepath_or_buffer, (str, Path)):
      path_str = str(filepath_or_buffer)
      if os.path.exists(path_str):
        with open(path_str, "r", encoding="utf-8") as f:
          data = json.load(f)
      else:
        data = json.loads(path_str)
    elif isinstance(filepath_or_buffer, dict):
      data = filepath_or_buffer
    elif hasattr(filepath_or_buffer, "read"):
      raw = filepath_or_buffer.read()
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
      data = json.loads(raw)
    else:
      raise TypeError(
          f"Unsupported type for from_json: {type(filepath_or_buffer)}"
      )

    model = cls(
        model_id=data.get("id", "Imported_Model"),
        name=data.get("name", "Imported Model"),
    )
    if "compartments" in data and isinstance(data["compartments"], dict):
      model.compartments = dict(data["compartments"])

    for m in data.get("metabolites", []):
      if isinstance(m, dict):
        model.add_metabolite(
            met_id=m["id"],
            name=m.get("name", m["id"]),
            compartment=m.get("compartment", ""),
            formula=m.get("formula", ""),
            charge=int(m.get("charge", 0)),
        )
      elif isinstance(m, str):
        model.add_metabolite(met_id=m, name=m)

    for r in data.get("reactions", []):
      if isinstance(r, dict):
        stoich = r.get("metabolites", {})
        lb = float(r.get("lower_bound", 0.0))
        ub = float(r.get("upper_bound", 1000.0))
        obj_coeff = float(r.get("objective_coefficient", 0.0))
        rxn_id = r["id"]
        model.add_reaction(
            rxn_id=rxn_id,
            name=r.get("name", rxn_id),
            stoichiometry=stoich,
            lower_bound=lb,
            upper_bound=ub,
            objective_coefficient=obj_coeff,
        )
        if obj_coeff > 0.0:
          model.objective_reaction = rxn_id

    return model

  def to_sbml(self, filepath: str) -> None:
    """Serializes model to SBML Level 3 Version 1 with FBC v2 XML."""
    root = ET.Element(
        "sbml",
        {
            "xmlns": "http://www.sbml.org/sbml/level3/version1/core",
            "xmlns:fbc": "http://www.sbml.org/sbml/level3/version1/fbc/version2",
            "level": "3",
            "version": "1",
            "fbc:required": "false",
        },
    )
    model_elem = ET.SubElement(
        root,
        "model",
        {
            "id": self.model_id,
            "name": self.name,
            "fbc:strict": "false",
        },
    )

    # Compartments
    cmps_elem = ET.SubElement(model_elem, "listOfCompartments")
    for c_id, c_name in self.compartments.items():
      ET.SubElement(
          cmps_elem, "compartment", {"id": c_id, "name": c_name, "constant": "true"}
      )

    # Species
    species_elem = ET.SubElement(model_elem, "listOfSpecies")
    for m in self.metabolites.values():
      s_id = f"M_{m.id.replace('[', '__').replace(']', '__')}"
      ET.SubElement(
          species_elem,
          "species",
          {
              "id": s_id,
              "name": m.id,
              "compartment": m.compartment or "r_shoot",
              "hasOnlySubstanceUnits": "false",
              "boundaryCondition": "false",
              "constant": "false",
              "fbc:chemicalFormula": m.formula,
          },
      )

    # Parameters (Bounds)
    params_elem = ET.SubElement(model_elem, "listOfParameters")
    for r in self.reactions.values():
      ET.SubElement(
          params_elem,
          "parameter",
          {
              "id": f"{r.id}_lb",
              "value": str(r.lower_bound),
              "constant": "true",
          },
      )
      ET.SubElement(
          params_elem,
          "parameter",
          {
              "id": f"{r.id}_ub",
              "value": str(r.upper_bound),
              "constant": "true",
          },
      )

    # Reactions
    rxns_elem = ET.SubElement(model_elem, "listOfReactions")
    for r in self.reactions.values():
      rxn_elem = ET.SubElement(
          rxns_elem,
          "reaction",
          {
              "id": f"R_{r.id}",
              "name": r.id,
              "reversible": "true" if r.lower_bound < 0 else "false",
              "fbc:lowerFluxBound": f"{r.id}_lb",
              "fbc:upperFluxBound": f"{r.id}_ub",
          },
      )
      reactants = {m: c for m, c in r.stoichiometry.items() if c < 0}
      products = {m: c for m, c in r.stoichiometry.items() if c > 0}

      if reactants:
        reactants_elem = ET.SubElement(rxn_elem, "listOfReactants")
        for m_id, coeff in reactants.items():
          s_id = f"M_{m_id.replace('[', '__').replace(']', '__')}"
          ET.SubElement(
              reactants_elem,
              "speciesReference",
              {
                  "species": s_id,
                  "stoichiometry": str(abs(coeff)),
                  "constant": "true",
              },
          )

      if products:
        products_elem = ET.SubElement(rxn_elem, "listOfProducts")
        for m_id, coeff in products.items():
          s_id = f"M_{m_id.replace('[', '__').replace(']', '__')}"
          ET.SubElement(
              products_elem,
              "speciesReference",
              {
                  "species": s_id,
                  "stoichiometry": str(abs(coeff)),
                  "constant": "true",
              },
          )

    # FBC Objectives
    obj_list_elem = ET.SubElement(
        model_elem,
        "fbc:listOfObjectives",
        {"fbc:activeObjective": f"obj_{self.objective_reaction}"},
    )
    obj_elem = ET.SubElement(
        obj_list_elem,
        "fbc:objective",
        {
            "fbc:id": f"obj_{self.objective_reaction}",
            "fbc:type": "maximize"
            if self.objective_direction == "max"
            else "minimize",
        },
    )
    flux_objs_elem = ET.SubElement(obj_elem, "fbc:listOfFluxObjectives")
    ET.SubElement(
        flux_objs_elem,
        "fbc:fluxObjective",
        {
            "fbc:reaction": f"R_{self.objective_reaction}",
            "fbc:coefficient": "1.0",
        },
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)

  @classmethod
  def from_sbml(cls, filepath_or_buffer: Any) -> TwoCompartmentMetabolicModel:
    """Deserializes model from SBML XML file path or string buffer."""
    if isinstance(filepath_or_buffer, (str, Path)):
      path_str = str(filepath_or_buffer)
      if os.path.exists(path_str):
        tree = ET.parse(path_str)
        root = tree.getroot()
      else:
        root = ET.fromstring(path_str)
    elif hasattr(filepath_or_buffer, "read"):
      raw = filepath_or_buffer.read()
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
      root = ET.fromstring(raw)
    else:
      raise TypeError(
          f"Unsupported type for from_sbml: {type(filepath_or_buffer)}"
      )

    def find_all_local(element, local_tag):
      return [e for e in element.iter() if e.tag.split("}")[-1] == local_tag]

    model_nodes = find_all_local(root, "model")
    if not model_nodes:
      raise ValueError("Invalid SBML: no <model> element found.")
    m_elem = model_nodes[0]

    model = cls(
        model_id=m_elem.get("id", "Imported_SBML"),
        name=m_elem.get("name", "Imported SBML Model"),
    )

    # Compartments
    for c_elem in find_all_local(m_elem, "compartment"):
      c_id = c_elem.get("id")
      if c_id:
        model.compartments[c_id] = c_elem.get("name", c_id)

    # Parameters (bounds map)
    params = {}
    for p_elem in find_all_local(m_elem, "parameter"):
      p_id = p_elem.get("id")
      p_val = p_elem.get("value")
      if p_id and p_val is not None:
        try:
          params[p_id] = float(p_val)
        except ValueError:
          pass

    # Species
    species_id_to_met = {}
    for s_elem in find_all_local(m_elem, "species"):
      s_id = s_elem.get("id")
      raw_name = s_elem.get("name") or s_id
      if "[" in raw_name and "]" in raw_name:
        met_id = raw_name
      else:
        cleaned = s_id
        if cleaned.startswith("M_"):
          cleaned = cleaned[2:]
        met_id = re.sub(r"__([a-zA-Z0-9_]+)__$", r"[\1]", cleaned)

      comp = s_elem.get("compartment", "")
      formula = s_elem.get("fbc:chemicalFormula", "") or s_elem.get(
          "{http://www.sbml.org/sbml/level3/version1/fbc/version2}chemicalFormula",
          "",
      )
      model.add_metabolite(
          met_id=met_id, name=raw_name, compartment=comp, formula=formula
      )
      species_id_to_met[s_id] = met_id

    # Reactions
    for r_elem in find_all_local(m_elem, "reaction"):
      r_id = r_elem.get("id", "")
      clean_r_id = r_elem.get("name") or (
          r_id[2:] if r_id.startswith("R_") else r_id
      )

      lb_attr = r_elem.get("fbc:lowerFluxBound") or r_elem.get(
          "{http://www.sbml.org/sbml/level3/version1/fbc/version2}lowerFluxBound"
      )
      ub_attr = r_elem.get("fbc:upperFluxBound") or r_elem.get(
          "{http://www.sbml.org/sbml/level3/version1/fbc/version2}upperFluxBound"
      )

      lb = (
          params.get(lb_attr, 0.0)
          if lb_attr in params
          else (float(lb_attr) if lb_attr else 0.0)
      )
      ub = (
          params.get(ub_attr, 1000.0)
          if ub_attr in params
          else (float(ub_attr) if ub_attr else 1000.0)
      )

      stoich = {}
      for r_ref in find_all_local(r_elem, "listOfReactants"):
        for s_ref in find_all_local(r_ref, "speciesReference"):
          sp = s_ref.get("species")
          s_val = float(s_ref.get("stoichiometry", 1.0))
          if sp in species_id_to_met:
            stoich[species_id_to_met[sp]] = -s_val

      for p_ref in find_all_local(r_elem, "listOfProducts"):
        for s_ref in find_all_local(p_ref, "speciesReference"):
          sp = s_ref.get("species")
          s_val = float(s_ref.get("stoichiometry", 1.0))
          if sp in species_id_to_met:
            stoich[species_id_to_met[sp]] = s_val

      model.add_reaction(
          rxn_id=clean_r_id,
          name=clean_r_id,
          stoichiometry=stoich,
          lower_bound=lb,
          upper_bound=ub,
      )

    # Active Objective
    for fo_elem in find_all_local(m_elem, "fluxObjective"):
      rxn_ref = fo_elem.get("fbc:reaction") or fo_elem.get(
          "{http://www.sbml.org/sbml/level3/version1/fbc/version2}reaction"
      )
      if rxn_ref:
        clean_obj = rxn_ref[2:] if rxn_ref.startswith("R_") else rxn_ref
        if clean_obj in model.reactions:
          model.objective_reaction = clean_obj

    return model


# ==============================================================================
# Model Factory and Universal Loader
# ==============================================================================


def build_two_compartment_model() -> TwoCompartmentMetabolicModel:
  """Constructs and returns the curated two-compartment metabolic model

  with 4 compartments, 18 internal metabolites, and 24 balanced reactions.
  """
  model = TwoCompartmentMetabolicModel()

  # 18 Metabolites
  model.add_metabolite(
      "photon[r_shoot]", "Solar Photon", "r_shoot", formula=""
  )
  model.add_metabolite(
      "co2[r_shoot]", "Carbon Dioxide", "r_shoot", formula="CO2"
  )
  model.add_metabolite("h2o[r_shoot]", "Water", "r_shoot", formula="H2O")
  model.add_metabolite("o2[r_shoot]", "Oxygen", "r_shoot", formula="O2")
  model.add_metabolite(
      "triose_p[r_shoot]",
      "Triose Phosphate",
      "r_shoot",
      formula="C3H7O6P",
      charge=-2,
  )
  model.add_metabolite(
      "sucrose[r_root]", "Root Sucrose", "r_root", formula="C12H22O11"
  )
  model.add_metabolite(
      "atp[r_root]", "Root ATP", "r_root", formula="C10H16N5O13P3", charge=-4
  )
  model.add_metabolite(
      "nh4[r_root]", "Root Ammonium", "r_root", formula="NH4", charge=1
  )
  model.add_metabolite(
      "amino_acid[r_root]",
      "Root Amino Acid Pool",
      "r_root",
      formula="C5H9NO4",
      charge=-1,
  )
  model.add_metabolite(
      "biomass[r_root]", "Rice Plant Biomass", "r_root", formula=""
  )
  model.add_metabolite(
      "sucrose[rhizo]", "Rhizosphere Sucrose", "rhizo", formula="C12H22O11"
  )
  model.add_metabolite(
      "nh4[rhizo]", "Rhizosphere Ammonium", "rhizo", formula="NH4", charge=1
  )
  model.add_metabolite(
      "n2[rhizo]", "Rhizosphere Dinitrogen", "rhizo", formula="N2"
  )
  model.add_metabolite(
      "h2[rhizo]", "Rhizosphere Hydrogen", "rhizo", formula="H2"
  )
  model.add_metabolite(
      "sucrose[a_bras]", "Bacterial Sucrose", "a_bras", formula="C12H22O11"
  )
  model.add_metabolite(
      "atp[a_bras]",
      "Bacterial ATP",
      "a_bras",
      formula="C10H16N5O13P3",
      charge=-4,
  )
  model.add_metabolite(
      "nh4[a_bras]", "Bacterial Ammonium", "a_bras", formula="NH4", charge=1
  )
  model.add_metabolite(
      "biomass[a_bras]", "Azospirillum Biomass", "a_bras", formula=""
  )

  # 24 Reactions
  # 1. EX_photon_r: -> photon[r_shoot] [0.0, 100.0]
  model.add_reaction(
      "EX_photon_r",
      "Solar Photon Influx",
      {"photon[r_shoot]": 1.0},
      0.0,
      100.0,
  )

  # 2. PHOTOSYN_r: 8 photon + 3 co2 + 3 h2o -> 1 triose_p + 3 o2 [0.0, 1000.0]
  model.add_reaction(
      "PHOTOSYN_r",
      "C3 Photosynthesis & Calvin Cycle",
      {
          "photon[r_shoot]": -8.0,
          "co2[r_shoot]": -3.0,
          "h2o[r_shoot]": -3.0,
          "triose_p[r_shoot]": 1.0,
          "o2[r_shoot]": 3.0,
      },
      0.0,
      1000.0,
  )

  # 3. SUC_SYN_r: 4 triose_p -> 1 sucrose[r_root] [0.0, 1000.0]
  model.add_reaction(
      "SUC_SYN_r",
      "Sucrose Synthesis & Phloem Translocation",
      {"triose_p[r_shoot]": -4.0, "sucrose[r_root]": 1.0},
      0.0,
      1000.0,
  )

  # 4. RESP_r: 1 sucrose[r_root] -> 30 atp[r_root] [0.0, 1000.0]
  model.add_reaction(
      "RESP_r",
      "Root Aerobic Respiration",
      {"sucrose[r_root]": -1.0, "atp[r_root]": 30.0},
      0.0,
      1000.0,
  )

  # 5. ATPM_r: 1 atp[r_root] -> [2.5, 1000.0]
  model.add_reaction(
      "ATPM_r", "Root Cellular Maintenance", {"atp[r_root]": -1.0}, 2.5, 1000.0
  )

  # 6. GS_GOGAT_r: 1 nh4[r_root] + 0.417 sucrose[r_root] + 1 atp[r_root] -> 1 amino_acid[r_root] [0.0, 1000.0]
  model.add_reaction(
      "GS_GOGAT_r",
      "Root Ammonium Assimilation",
      {
          "nh4[r_root]": -1.0,
          "sucrose[r_root]": -0.417,
          "atp[r_root]": -1.0,
          "amino_acid[r_root]": 1.0,
      },
      0.0,
      1000.0,
  )

  # 7. BIOMASS_rice: 0.25 sucrose + 0.15 amino_acid + 5 atp -> 1 biomass[r_root] [0.0, 1000.0]
  model.add_reaction(
      "BIOMASS_rice",
      "Rice Host Vegetative Biomass",
      {
          "sucrose[r_root]": -0.25,
          "amino_acid[r_root]": -0.15,
          "atp[r_root]": -5.0,
          "biomass[r_root]": 1.0,
      },
      0.0,
      1000.0,
      objective_coefficient=1.0,
  )

  # 8. SWEET11_export: 1 sucrose[r_root] -> 1 sucrose[rhizo] [0.0, 10.0] (WT)
  model.add_reaction(
      "SWEET11_export",
      "OsSWEET11 Sucrose Transporter",
      {"sucrose[r_root]": -1.0, "sucrose[rhizo]": 1.0},
      0.0,
      10.0,
  )

  # 9. EX_nh4_soil: -> 1 nh4[rhizo] [0.0, 5.0]
  model.add_reaction(
      "EX_nh4_soil",
      "External Soil Ammonium Influx",
      {"nh4[rhizo]": 1.0},
      0.0,
      5.0,
  )

  # 10. NH4_uptake_rice: 1 nh4[rhizo] -> 1 nh4[r_root] [0.0, 1000.0]
  model.add_reaction(
      "NH4_uptake_rice",
      "Rice Root Ammonium Transporter",
      {"nh4[rhizo]": -1.0, "nh4[r_root]": 1.0},
      0.0,
      1000.0,
  )

  # 11. SUC_uptake_ab: 1 sucrose[rhizo] -> 1 sucrose[a_bras] [0.0, 1000.0]
  model.add_reaction(
      "SUC_uptake_ab",
      "Bacterial Sucrose Influx",
      {"sucrose[rhizo]": -1.0, "sucrose[a_bras]": 1.0},
      0.0,
      1000.0,
  )

  # 12. RESP_ab: 1 sucrose[a_bras] -> 20 atp[a_bras] [0.0, 1000.0]
  model.add_reaction(
      "RESP_ab",
      "Bacterial Microaerophilic Respiration",
      {"sucrose[a_bras]": -1.0, "atp[a_bras]": 20.0},
      0.0,
      1000.0,
  )

  # 13. ATPM_ab: 1 atp[a_bras] -> [0.5, 1000.0]
  model.add_reaction(
      "ATPM_ab",
      "Bacterial Non-Growth Maintenance",
      {"atp[a_bras]": -1.0},
      0.5,
      1000.0,
  )

  # 14. NITROGENASE_ab: 1 n2[rhizo] + 16 atp[a_bras] -> 2 nh4[a_bras] + 1 h2[rhizo] [0.0, 1000.0]
  model.add_reaction(
      "NITROGENASE_ab",
      "Diazotrophic Mo-Fe Nitrogenase",
      {
          "n2[rhizo]": -1.0,
          "atp[a_bras]": -16.0,
          "nh4[a_bras]": 2.0,
          "h2[rhizo]": 1.0,
      },
      0.0,
      1000.0,
  )

  # 15. NH4_excretion_ab: 1 nh4[a_bras] -> 1 nh4[rhizo] [0.0, 1000.0]
  model.add_reaction(
      "NH4_excretion_ab",
      "Symbiotic Ammonium Excretion",
      {"nh4[a_bras]": -1.0, "nh4[rhizo]": 1.0},
      0.0,
      1000.0,
  )

  # 16. NH4_uptake_ab: 1 nh4[rhizo] -> 1 nh4[a_bras] [0.0, 1000.0]
  model.add_reaction(
      "NH4_uptake_ab",
      "Direct Bacterial Soil NH4 Uptake",
      {"nh4[rhizo]": -1.0, "nh4[a_bras]": 1.0},
      0.0,
      1000.0,
  )

  # 17. BIOMASS_ab: 0.1 sucrose + 0.5 nh4 + 4 atp -> 1 biomass[a_bras] [0.0, 1000.0]
  model.add_reaction(
      "BIOMASS_ab",
      "Bacterial Biomass Synthesis",
      {
          "sucrose[a_bras]": -0.1,
          "nh4[a_bras]": -0.5,
          "atp[a_bras]": -4.0,
          "biomass[a_bras]": 1.0,
      },
      0.0,
      1000.0,
  )

  # 18. EX_co2_r: -> co2[r_shoot] [0.0, 1000.0]
  model.add_reaction(
      "EX_co2_r",
      "Atmospheric CO2 Exchange",
      {"co2[r_shoot]": 1.0},
      0.0,
      1000.0,
  )

  # 19. EX_h2o_r: -> h2o[r_shoot] [0.0, 1000.0]
  model.add_reaction(
      "EX_h2o_r", "Plant Water Influx", {"h2o[r_shoot]": 1.0}, 0.0, 1000.0
  )

  # 20. EX_o2_r: o2[r_shoot] -> [0.0, 1000.0]
  model.add_reaction(
      "EX_o2_r", "Foliar Oxygen Efflux", {"o2[r_shoot]": -1.0}, 0.0, 1000.0
  )

  # 21. EX_n2_rhizo: -> n2[rhizo] [0.0, 1000.0]
  model.add_reaction(
      "EX_n2_rhizo", "Soil N2 Gas Aeration", {"n2[rhizo]": 1.0}, 0.0, 1000.0
  )

  # 22. EX_h2_rhizo: h2[rhizo] -> [0.0, 1000.0]
  model.add_reaction(
      "EX_h2_rhizo", "Hydrogen Gas Efflux", {"h2[rhizo]": -1.0}, 0.0, 1000.0
  )

  # 23. SK_biomass_rice: biomass[r_root] -> [0.0, 1000.0]
  model.add_reaction(
      "SK_biomass_rice",
      "Rice Biomass Accumulation Sink",
      {"biomass[r_root]": -1.0},
      0.0,
      1000.0,
  )

  # 24. SK_biomass_ab: biomass[a_bras] -> [0.0, 1000.0]
  model.add_reaction(
      "SK_biomass_ab",
      "Bacterial Biomass Accumulation Sink",
      {"biomass[a_bras]": -1.0},
      0.0,
      1000.0,
  )

  return model


def load_model_from_file(filepath_or_buffer: Any) -> TwoCompartmentMetabolicModel:
  """Universal loader that auto-detects SBML (.xml, .sbml) or JSON (.json)

  from file path, file buffer, or string content.
  """
  if isinstance(filepath_or_buffer, (str, Path)):
    path_str = str(filepath_or_buffer).strip()
    if os.path.exists(path_str):
      lower_path = path_str.lower()
      if lower_path.endswith((".json",)):
        return TwoCompartmentMetabolicModel.from_json(path_str)
      elif lower_path.endswith((".xml", ".sbml")):
        return TwoCompartmentMetabolicModel.from_sbml(path_str)
      else:
        with open(path_str, "r", encoding="utf-8") as f:
          head = f.read(256).strip()
        if head.startswith("<") or "<?xml" in head or "<sbml" in head:
          return TwoCompartmentMetabolicModel.from_sbml(path_str)
        else:
          return TwoCompartmentMetabolicModel.from_json(path_str)
    else:
      head = path_str[:256].strip()
      if head.startswith("<") or "<?xml" in head or "<sbml" in head:
        return TwoCompartmentMetabolicModel.from_sbml(path_str)
      else:
        return TwoCompartmentMetabolicModel.from_json(path_str)
  else:
    content = filepath_or_buffer.read()
    if isinstance(content, bytes):
      content_str = content.decode("utf-8")
    else:
      content_str = str(content)
    if hasattr(filepath_or_buffer, "seek"):
      filepath_or_buffer.seek(0)
    head = content_str[:256].strip()
    if head.startswith("<") or "<?xml" in head or "<sbml" in head:
      return TwoCompartmentMetabolicModel.from_sbml(content_str)
    else:
      return TwoCompartmentMetabolicModel.from_json(content_str)
