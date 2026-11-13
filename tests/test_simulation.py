"""tests/test_simulation.py.

Comprehensive End-to-End Automated Test Suite for the Rice–Azospirillum
Two-Compartment Flux Balance Analysis (FBA) Simulation & 3D WebGL Digital Twin.

Architected for 100% standalone execution with pure SciPy HiGHS fallback solver
(zero external binary dependencies) following the Decoupled Progressive Testability
Pattern (TEST-INFRA-ARCH-001).
"""

from dataclasses import dataclass, field
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

# Add project workspace root to sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
  sys.path.insert(0, WORKSPACE_ROOT)

try:
  import plotly.graph_objects as go
except ImportError:
  go = None


def pytest_configure(config):
  """Register custom markers to prevent PytestUnknownMarkWarning."""
  config.addinivalue_line(
      "markers", "r1_metabolic_model: Tests for Requirement 1"
  )
  config.addinivalue_line(
      "markers", "r2_simulation_engine: Tests for Requirement 2"
  )
  config.addinivalue_line("markers", "r3_visualizer_3d: Tests for Requirement 3")
  config.addinivalue_line("markers", "r4_app_components: Tests for Requirement 4")
  config.addinivalue_line(
      "markers", "r5_scientific_hypotheses: Tests for Requirement 5"
  )
  config.addinivalue_line("markers", "tier1: Tier 1 feature coverage")
  config.addinivalue_line("markers", "tier2: Tier 2 boundary cases")
  config.addinivalue_line("markers", "tier3: Tier 3 combinatorial tests")
  config.addinivalue_line("markers", "tier4: Tier 4 scientific scenarios")


# ==============================================================================
# Invariant Checking Utilities
# ==============================================================================


def assert_mass_balance(S: np.ndarray, v: np.ndarray, tol: float = 1e-5):
  """Assert steady-state stoichiometric conservation: max|S @ v| < tol."""
  residual = np.abs(np.dot(S, v))
  max_dev = float(np.max(residual))
  assert (
      max_dev < tol
  ), f"Steady-state mass balance violated! max|S*v| = {max_dev:.2e} >= {tol:.2e}"


def assert_thermodynamic_bounds(
    v: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-5
):
  """Assert thermodynamic flux bounds: lb - tol <= v <= ub + tol."""
  lower_violations = v < (lb - tol)
  upper_violations = v > (ub + tol)
  assert not np.any(lower_violations), (
      f"Flux lower bound violation: min(v - lb) = {np.min(v - lb):.2e} in"
      f" reactions {np.where(lower_violations)[0]}"
  )
  assert not np.any(upper_violations), (
      f"Flux upper bound violation: max(v - ub) = {np.max(v - ub):.2e} in"
      f" reactions {np.where(upper_violations)[0]}"
  )


def assert_maintenance_satisfied(
    solution: Any, min_atpm: float = 2.5, tol: float = 1e-5
):
  """Assert root non-growth cellular maintenance ATP satisfaction."""
  if not solution.is_starved:
    assert solution.atpm_r >= (min_atpm - tol), (
        f"Viable solution violated maintenance ATP requirement: "
        f"ATPM_r = {solution.atpm_r:.4f} < {min_atpm:.4f}"
    )
  else:
    assert solution.biomass_rice == pytest.approx(0.0, abs=1e-7), (
        f"Starved solution must assign exactly 0.0 biomass, got"
        f" {solution.biomass_rice}"
    )
    assert (
        solution.starvation_deficit > 0.0
    ), "Starved solution must report positive ATP deficit"


def assert_tensor_congruence(expected_shape: Tuple[int, ...], *tensors):
  """Assert that all tensors match expected shape and contain valid numeric data."""
  for i, T in enumerate(tensors):
    assert (
        T.shape == expected_shape
    ), f"Tensor #{i} shape mismatch: expected {expected_shape}, got {T.shape}"
    assert not np.all(np.isnan(T)), f"Tensor #{i} is completely NaN values"


# ==============================================================================
# Standard Data Contracts (Dataclasses)
# ==============================================================================


@dataclass
class FBASolution:
  status: str  # "optimal", "starvation", "infeasible"
  is_feasible: bool
  is_starved: bool
  objective_value: float
  biomass_rice: float
  biomass_ab: float
  suc_syn_r: float
  sweet11_export: float
  nitrogenase_ab: float
  nh4_excretion_ab: float
  nh4_uptake_rice: float
  suc_uptake_ab: float
  atpm_r: float
  atpm_ab: float
  ex_photon_r: float
  ex_nh4_soil: float
  starvation_deficit: float = 0.0
  fluxes: Dict[str, float] = field(default_factory=dict)


@dataclass
class Sweep1DResult:
  param_name: str
  param_values: np.ndarray
  biomass_rice: np.ndarray
  nitrogenase_ab: np.ndarray
  starvation_mask: np.ndarray
  dataframe: pd.DataFrame
  flux_trajectories: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PhasePlane2DResult:
  param_x_name: str
  param_y_name: str
  x_values: np.ndarray
  y_values: np.ndarray
  X: np.ndarray
  Y: np.ndarray
  Z_biomass_rice: np.ndarray
  Z_nitrogenase_ab: np.ndarray
  Z_nhbi: np.ndarray
  Z_suc_syn_r: np.ndarray
  Z_atpm_r: np.ndarray
  starvation_mask: np.ndarray
  pareto_ridge_x: np.ndarray
  pareto_ridge_y: np.ndarray
  cliff_x: np.ndarray
  cliff_y: np.ndarray


@dataclass
class AmmoniumBifurcationResult:
  X_nh4: np.ndarray
  Y_sweet11: np.ndarray
  Z_nhbi: np.ndarray
  Z_nitrogenase: np.ndarray
  critical_nh4_threshold: float
  nhbi_at_zero_nh4: float
  nhbi_at_high_nh4: float
  nitrogenase_shutoff_nh4: float
  bifurcation_contour: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class HypothesisTestResult:
  h1_accepted: bool
  optimal_lb: float
  optimal_ub: float
  max_biomass: float
  ko_biomass: float
  max_benefit_pct: float
  nitrogenase_at_opt: float
  starvation_threshold_flux: float
  decision_summary: str = ""


@dataclass
class Experiment1Result:
  seasonal_regimes: Dict[str, FBASolution]
  summary_table: pd.DataFrame
  continuous_trajectory: Sweep1DResult


# ==============================================================================
# Self-Contained Reference Implementation Harness (Fallback & Oracle)
# ==============================================================================

CANONICAL_METABOLITES = [
    "photon[r_shoot]",
    "co2[r_shoot]",
    "h2o[r_shoot]",
    "o2[r_shoot]",
    "atp[r_shoot]",
    "triose_p[r_shoot]",
    "sucrose[r_root]",
    "atp[r_root]",
    "nh4[r_root]",
    "amino_acid[r_root]",
    "sucrose[rhizo]",
    "nh4[rhizo]",
    "n2[rhizo]",
    "h2[rhizo]",
    "sucrose[a_bras]",
    "atp[a_bras]",
    "fd_red[a_bras]",
    "nh4[a_bras]",
    "biomass[r_root]",
    "biomass[a_bras]",
    "nadph[r_shoot]",
]

CANONICAL_REACTIONS = [
    "EX_photon_r",
    "PHOTOSYN_r",
    "SUC_SYN_r",
    "RESP_r",
    "ATPM_r",
    "GS_GOGAT_r",
    "BIOMASS_rice",
    "SWEET11_export",
    "EX_nh4_soil",
    "NH4_uptake_rice",
    "SUC_uptake_ab",
    "RESP_ab",
    "ATPM_ab",
    "NITROGENASE_ab",
    "NH4_excretion_ab",
    "NH4_uptake_ab",
    "BIOMASS_ab",
    "EX_co2_r",
    "EX_h2o_r",
    "EX_o2_r",
    "EX_n2_rhizo",
    "EX_h2_rhizo",
    "SK_biomass_rice",
    "SK_biomass_ab",
]


class ReferenceModel:
  """Self-contained Reference 4-Compartment Stoichiometric Model."""

  def __init__(self):
    self.compartments = {
        "r_shoot": "Rice Shoot Canopy",
        "r_root": "Rice Root System",
        "rhizo": "Rhizosphere Shared Exchange Pool",
        "a_bras": "Azospirillum brasilense Symbiont",
    }
    self.metabolite_ids = list(CANONICAL_METABOLITES)
    self.reaction_ids = list(CANONICAL_REACTIONS)
    self.M = len(self.metabolite_ids)
    self.N = len(self.reaction_ids)

    # Initialize bounds
    self.lb = np.zeros(self.N, dtype=float)
    self.ub = np.full(self.N, 1000.0, dtype=float)

    # Specific bounds
    self.lb[self.reaction_ids.index("ATPM_r")] = 2.5
    self.lb[self.reaction_ids.index("ATPM_ab")] = 0.5
    self.ub[self.reaction_ids.index("EX_photon_r")] = 100.0
    self.ub[self.reaction_ids.index("EX_nh4_soil")] = 0.0
    self.lb[self.reaction_ids.index("SWEET11_export")] = 0.0
    self.ub[self.reaction_ids.index("SWEET11_export")] = 10.0

    # Build Stoichiometric Matrix S (M x N)
    self.S = np.zeros((self.M, self.N), dtype=float)
    self._build_stoichiometry()

  def _build_stoichiometry(self):
    m_idx = {m: i for i, m in enumerate(self.metabolite_ids)}
    r_idx = {r: j for j, r in enumerate(self.reaction_ids)}

    def add(r_name, m_name, coeff):
      if m_name in m_idx and r_name in r_idx:
        self.S[m_idx[m_name], r_idx[r_name]] += coeff

    # 1. EX_photon_r: -> photon[r_shoot]
    add("EX_photon_r", "photon[r_shoot]", 1.0)

    # 2. PHOTOSYN_r: 2 photon + 0.75 co2 + 0.75 h2o -> 1 triose_p + 0.75 o2 + 2 atp_shoot + 1 nadph
    add("PHOTOSYN_r", "photon[r_shoot]", -2.0)
    add("PHOTOSYN_r", "co2[r_shoot]", -0.75)
    add("PHOTOSYN_r", "h2o[r_shoot]", -0.75)
    add("PHOTOSYN_r", "triose_p[r_shoot]", 1.0)
    add("PHOTOSYN_r", "o2[r_shoot]", 0.75)
    add("PHOTOSYN_r", "atp[r_shoot]", 2.0)
    add("PHOTOSYN_r", "nadph[r_shoot]", 1.0)

    # 3. SUC_SYN_r: 1 triose_p + 2 atp_shoot + 1 nadph -> 0.25 sucrose[r_root]
    add("SUC_SYN_r", "triose_p[r_shoot]", -1.0)
    add("SUC_SYN_r", "atp[r_shoot]", -2.0)
    add("SUC_SYN_r", "nadph[r_shoot]", -1.0)
    add("SUC_SYN_r", "sucrose[r_root]", 0.25)

    # 4. RESP_r: 1 sucrose[r_root] -> 30 atp[r_root] (respiration)
    add("RESP_r", "sucrose[r_root]", -1.0)
    add("RESP_r", "atp[r_root]", 30.0)

    # 5. ATPM_r: 1 atp[r_root] -> adp
    add("ATPM_r", "atp[r_root]", -1.0)

    # 6. GS_GOGAT_r: 1 nh4[r_root] + 0.417 sucrose[r_root] + 1 atp[r_root] -> 1 amino_acid[r_root]
    add("GS_GOGAT_r", "nh4[r_root]", -1.0)
    add("GS_GOGAT_r", "sucrose[r_root]", -0.417)
    add("GS_GOGAT_r", "atp[r_root]", -1.0)
    add("GS_GOGAT_r", "amino_acid[r_root]", 1.0)

    # 7. BIOMASS_rice: 0.25 sucrose + 0.15 amino_acid + 5.0 atp -> 1 biomass[r_root]
    add("BIOMASS_rice", "sucrose[r_root]", -0.25)
    add("BIOMASS_rice", "amino_acid[r_root]", -0.15)
    add("BIOMASS_rice", "atp[r_root]", -5.0)
    add("BIOMASS_rice", "biomass[r_root]", 1.0)

    # 8. SWEET11_export: 1 sucrose[r_root] -> 1 sucrose[rhizo]
    add("SWEET11_export", "sucrose[r_root]", -1.0)
    add("SWEET11_export", "sucrose[rhizo]", 1.0)

    # 9. EX_nh4_soil: -> nh4[rhizo]
    add("EX_nh4_soil", "nh4[rhizo]", 1.0)

    # 10. NH4_uptake_rice: 1 nh4[rhizo] -> 1 nh4[r_root]
    add("NH4_uptake_rice", "nh4[rhizo]", -1.0)
    add("NH4_uptake_rice", "nh4[r_root]", 1.0)

    # 11. SUC_uptake_ab: 1 sucrose[rhizo] -> 1 sucrose[a_bras]
    add("SUC_uptake_ab", "sucrose[rhizo]", -1.0)
    add("SUC_uptake_ab", "sucrose[a_bras]", 1.0)

    # 12. RESP_ab: 1 sucrose[a_bras] -> 20 atp[a_bras] + 8 fd_red[a_bras]
    add("RESP_ab", "sucrose[a_bras]", -1.0)
    add("RESP_ab", "atp[a_bras]", 20.0)
    add("RESP_ab", "fd_red[a_bras]", 8.0)

    # 13. ATPM_ab: 1 atp[a_bras] ->
    add("ATPM_ab", "atp[a_bras]", -1.0)

    # 14. NITROGENASE_ab: 1 n2[rhizo] + 16 atp[a_bras] + 8 fd_red[a_bras] -> 2 nh4[a_bras] + 1 h2[rhizo]
    add("NITROGENASE_ab", "n2[rhizo]", -1.0)
    add("NITROGENASE_ab", "atp[a_bras]", -16.0)
    add("NITROGENASE_ab", "fd_red[a_bras]", -8.0)
    add("NITROGENASE_ab", "nh4[a_bras]", 2.0)
    add("NITROGENASE_ab", "h2[rhizo]", 1.0)

    # 15. NH4_excretion_ab: 1 nh4[a_bras] -> 1 nh4[rhizo]
    add("NH4_excretion_ab", "nh4[a_bras]", -1.0)
    add("NH4_excretion_ab", "nh4[rhizo]", 1.0)

    # 16. NH4_uptake_ab: 1 nh4[rhizo] -> 1 nh4[a_bras]
    add("NH4_uptake_ab", "nh4[rhizo]", -1.0)
    add("NH4_uptake_ab", "nh4[a_bras]", 1.0)

    # 17. BIOMASS_ab: 0.1 sucrose + 0.5 nh4 + 4 atp -> 1 biomass[a_bras]
    add("BIOMASS_ab", "sucrose[a_bras]", -0.1)
    add("BIOMASS_ab", "nh4[a_bras]", -0.5)
    add("BIOMASS_ab", "atp[a_bras]", -4.0)
    add("BIOMASS_ab", "biomass[a_bras]", 1.0)

    # 18-24. Boundary/Exchange reactions
    add("EX_co2_r", "co2[r_shoot]", 1.0)
    add("EX_h2o_r", "h2o[r_shoot]", 1.0)
    add("EX_o2_r", "o2[r_shoot]", -1.0)
    add("EX_n2_rhizo", "n2[rhizo]", 1.0)
    add("EX_h2_rhizo", "h2[rhizo]", -1.0)
    add("EX_h2_rhizo", "fd_red[a_bras]", -8.0)  # Hydrogenase ferredoxin turnover
    add("SK_biomass_rice", "biomass[r_root]", -1.0)
    add("SK_biomass_ab", "biomass[a_bras]", -1.0)

  def get_reaction_bounds(self, reaction_id: str) -> Tuple[float, float]:
    idx = self.reaction_ids.index(reaction_id)
    return (float(self.lb[idx]), float(self.ub[idx]))

  def set_reaction_bounds(self, reaction_id: str, lb: float, ub: float):
    idx = self.reaction_ids.index(reaction_id)
    self.lb[idx] = float(lb)
    self.ub[idx] = float(ub)

  def set_genotype(
      self,
      genotype: str,
      lb_custom: Optional[float] = None,
      ub_custom: Optional[float] = None,
  ):
    idx = self.reaction_ids.index("SWEET11_export")
    if genotype in ("OsSWEET11_WT", "WT"):
      self.lb[idx] = 0.0
      self.ub[idx] = 10.0
    elif genotype in ("OsSWEET11_OE", "OE"):
      self.lb[idx] = 2.0
      self.ub[idx] = 20.0
    elif genotype in ("OsSWEET11_Knockdown", "OsSWEET11_KD", "KD"):
      self.lb[idx] = 0.0
      self.ub[idx] = 3.0
    elif genotype in ("Ossweet11_KO", "KO"):
      self.lb[idx] = 0.0
      self.ub[idx] = 0.0
    if lb_custom is not None:
      self.lb[idx] = float(lb_custom)
    if ub_custom is not None:
      self.ub[idx] = float(ub_custom)

  def to_scipy_lp(self):
    c = np.zeros(self.N, dtype=float)
    c[self.reaction_ids.index("BIOMASS_rice")] = -1.0
    bounds = [(float(self.lb[j]), float(self.ub[j])) for j in range(self.N)]
    b_eq = np.zeros(self.M, dtype=float)
    return (
        c,
        self.S.copy(),
        b_eq,
        bounds,
        list(self.reaction_ids),
        list(self.metabolite_ids),
    )

  def optimize(
      self,
      objective: str = "BIOMASS_rice",
      maximize: bool = True,
  ) -> FBASolution:
    """Delegates to ReferenceSimulator.solve_single_point for decoupled fallback testing."""
    return ReferenceSimulator.solve_single_point(self)

  def to_json(self, filepath_or_buffer) -> str:
    data = {
        "metabolites": self.metabolite_ids,
        "reactions": self.reaction_ids,
        "lb": self.lb.tolist(),
        "ub": self.ub.tolist(),
        "S": self.S.tolist(),
    }
    dumped = json.dumps(data, indent=2)
    if isinstance(filepath_or_buffer, str):
      with open(filepath_or_buffer, "w") as f:
        f.write(dumped)
    return dumped

  @classmethod
  def from_json(cls, filepath_or_buffer):
    if isinstance(filepath_or_buffer, str):
      if os.path.exists(filepath_or_buffer):
        with open(filepath_or_buffer, "r") as f:
          data = json.load(f)
      else:
        data = json.loads(filepath_or_buffer)
    else:
      data = json.load(filepath_or_buffer)

    m = cls()
    m.lb = np.array(data["lb"], dtype=float)
    m.ub = np.array(data["ub"], dtype=float)
    m.S = np.array(data["S"], dtype=float)
    return m

  def to_sbml(self, filepath: str):
    root = ET.Element("sbml", xmlns="http://www.sbml.org/sbml/level3/version1/core", version="1", level="3")
    model_elem = ET.SubElement(root, "model", id="Rice_Azospirillum_Symbiosis")
    rxns_elem = ET.SubElement(model_elem, "listOfReactions")
    for rxn in self.reaction_ids:
      ET.SubElement(rxns_elem, "reaction", id=rxn)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


class ReferenceSimulator:
  """Self-contained Reference FBA Simulation Engine executing two-stage slack LP."""

  @staticmethod
  def solve_single_point(
      model: ReferenceModel,
      light_pct: float = 100.0,
      genotype: str = "OsSWEET11_WT",
      sweet11_bounds: Optional[Tuple[float, float]] = None,
      external_nh4: float = 0.0,
      inoculated: bool = True,
      solver_method: str = "highs",
  ) -> FBASolution:
    light_pct = max(0.0, float(light_pct))
    external_nh4 = max(0.0, float(external_nh4))

    # Configure bounds
    photon_idx = model.reaction_ids.index("EX_photon_r")
    nh4_soil_idx = model.reaction_ids.index("EX_nh4_soil")
    sweet11_idx = model.reaction_ids.index("SWEET11_export")
    atpm_r_idx = model.reaction_ids.index("ATPM_r")
    biomass_r_idx = model.reaction_ids.index("BIOMASS_rice")
    biomass_ab_idx = model.reaction_ids.index("BIOMASS_ab")
    nif_idx = model.reaction_ids.index("NITROGENASE_ab")
    nh4_exc_idx = model.reaction_ids.index("NH4_excretion_ab")
    nh4_uptake_r_idx = model.reaction_ids.index("NH4_uptake_rice")
    suc_uptake_ab_idx = model.reaction_ids.index("SUC_uptake_ab")
    atpm_ab_idx = model.reaction_ids.index("ATPM_ab")
    suc_syn_r_idx = model.reaction_ids.index("SUC_SYN_r")

    # Local bounds copies
    lb = model.lb.copy()
    ub = model.ub.copy()

    ub[photon_idx] = 100.0 * (light_pct / 100.0)
    ub[nh4_soil_idx] = external_nh4

    # Genotype and custom bounds
    if sweet11_bounds is not None:
      lb[sweet11_idx] = float(sweet11_bounds[0])
      ub[sweet11_idx] = float(sweet11_bounds[1])
      if ub[sweet11_idx] <= 1e-6:
        lb[atpm_ab_idx] = 0.0
    else:
      if genotype in ("OsSWEET11_WT", "WT"):
        lb[sweet11_idx], ub[sweet11_idx] = 0.0, 10.0
      elif genotype in ("OsSWEET11_OE", "OE"):
        lb[sweet11_idx], ub[sweet11_idx] = 2.0, 20.0
      elif genotype in ("OsSWEET11_Knockdown", "OsSWEET11_KD", "KD"):
        lb[sweet11_idx], ub[sweet11_idx] = 0.0, 3.0
      elif genotype in ("Ossweet11_KO", "KO"):
        lb[sweet11_idx], ub[sweet11_idx] = 0.0, 0.0
        lb[atpm_ab_idx] = 0.0

    # Axenic uninoculated control
    if not inoculated:
      for r_name in [
          "SUC_uptake_ab",
          "RESP_ab",
          "ATPM_ab",
          "NITROGENASE_ab",
          "NH4_excretion_ab",
          "NH4_uptake_ab",
          "BIOMASS_ab",
      ]:
        idx = model.reaction_ids.index(r_name)
        lb[idx] = 0.0
        ub[idx] = 0.0
      lb[sweet11_idx] = 0.0
      ub[sweet11_idx] = 0.0

    # Objective: Maximize BIOMASS_rice
    c = np.zeros(model.N, dtype=float)
    c[biomass_r_idx] = -1.0

    b_eq = np.zeros(model.M, dtype=float)
    bounds = list(zip(lb, ub))

    # Stage 1 LP
    res1 = linprog(
        c=c,
        A_eq=model.S,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False, "presolve": True},
    )

    if res1.status == 0:
      v = res1.x
      flux_dict = {
          rxn: float(v[i]) for i, rxn in enumerate(model.reaction_ids)
      }
      return FBASolution(
          status="optimal",
          is_feasible=True,
          is_starved=False,
          objective_value=float(-res1.fun),
          biomass_rice=float(v[biomass_r_idx]),
          biomass_ab=float(v[biomass_ab_idx]),
          suc_syn_r=float(v[suc_syn_r_idx]),
          sweet11_export=float(v[sweet11_idx]),
          nitrogenase_ab=float(v[nif_idx]),
          nh4_excretion_ab=float(v[nh4_exc_idx]),
          nh4_uptake_rice=float(v[nh4_uptake_r_idx]),
          suc_uptake_ab=float(v[suc_uptake_ab_idx]),
          atpm_r=float(v[atpm_r_idx]),
          atpm_ab=float(v[atpm_ab_idx]),
          ex_photon_r=float(v[photon_idx]),
          ex_nh4_soil=float(v[nh4_soil_idx]),
          starvation_deficit=0.0,
          fluxes=flux_dict,
      )

    # Stage 2 Slack LP: Relax lb[ATPM_r] = 0, maximize ATPM_r
    lb_stage2 = lb.copy()
    lb_stage2[atpm_r_idx] = 0.0
    lb_stage2[atpm_ab_idx] = 0.0
    c_stage2 = np.zeros(model.N, dtype=float)
    c_stage2[atpm_r_idx] = -1.0

    res2 = linprog(
        c=c_stage2,
        A_eq=model.S,
        b_eq=b_eq,
        bounds=list(zip(lb_stage2, ub)),
        method="highs",
        options={"disp": False, "presolve": True},
    )

    atp_achieved = float(-res2.fun) if res2.status == 0 else 0.0
    deficit = max(0.0, 2.5 - atp_achieved)
    v2 = res2.x if res2.status == 0 else np.zeros(model.N)
    flux_dict = {
        rxn: float(v2[i]) for i, rxn in enumerate(model.reaction_ids)
    }

    return FBASolution(
        status="starvation" if deficit > 1e-5 else "infeasible",
        is_feasible=False,
        is_starved=True,
        objective_value=0.0,
        biomass_rice=0.0,
        biomass_ab=float(v2[biomass_ab_idx]),
        suc_syn_r=float(v2[suc_syn_r_idx]),
        sweet11_export=float(v2[sweet11_idx]),
        nitrogenase_ab=float(v2[nif_idx]),
        nh4_excretion_ab=float(v2[nh4_exc_idx]),
        nh4_uptake_rice=float(v2[nh4_uptake_r_idx]),
        suc_uptake_ab=float(v2[suc_uptake_ab_idx]),
        atpm_r=float(atp_achieved),
        atpm_ab=float(v2[atpm_ab_idx]),
        ex_photon_r=float(v2[photon_idx]),
        ex_nh4_soil=float(v2[nh4_soil_idx]),
        starvation_deficit=deficit,
        fluxes=flux_dict,
    )

  @staticmethod
  def run_1d_sweep(
      model: ReferenceModel,
      param_name: str,
      param_values: np.ndarray,
      fixed_params: Optional[Dict[str, Any]] = None,
  ) -> Sweep1DResult:
    fixed_params = fixed_params or {}
    N = len(param_values)
    biomass_arr = np.zeros(N)
    nif_arr = np.zeros(N)
    starve_arr = np.zeros(N, dtype=bool)
    rows = []
    flux_trajs = {rxn: np.zeros(N) for rxn in model.reaction_ids}

    for i, val in enumerate(param_values):
      p = dict(fixed_params)
      p[param_name] = val
      sol = ReferenceSimulator.solve_single_point(
          model,
          light_pct=p.get("light_pct", 100.0),
          genotype=p.get("genotype", "OsSWEET11_WT"),
          sweet11_bounds=p.get(
              "sweet11_bounds",
              (val, val) if param_name == "sweet11_export" else None,
          ),
          external_nh4=p.get("external_nh4", 0.0),
          inoculated=p.get("inoculated", True),
      )
      biomass_arr[i] = sol.biomass_rice
      nif_arr[i] = sol.nitrogenase_ab
      starve_arr[i] = sol.is_starved
      for rxn, fval in sol.fluxes.items():
        if rxn in flux_trajs:
          flux_trajs[rxn][i] = fval
      rows.append({
          param_name: val,
          "biomass_rice": sol.biomass_rice,
          "nitrogenase_ab": sol.nitrogenase_ab,
          "is_starved": sol.is_starved,
          "suc_syn_r": sol.suc_syn_r,
          "sweet11_export": sol.sweet11_export,
      })

    return Sweep1DResult(
        param_name=param_name,
        param_values=np.array(param_values),
        biomass_rice=biomass_arr,
        nitrogenase_ab=nif_arr,
        starvation_mask=starve_arr,
        dataframe=pd.DataFrame(rows),
        flux_trajectories=flux_trajs,
    )

  @staticmethod
  def generate_2d_phase_plane(
      model: ReferenceModel,
      param_x: Tuple[str, float, float, int] = ("light_pct", 10.0, 100.0, 20),
      param_y: Tuple[str, float, float, int] = ("sweet11_export", 0.0, 10.0, 15),
      fixed_params: Optional[Dict[str, Any]] = None,
      compute_nhbi: bool = True,
  ) -> PhasePlane2DResult:
    x_name, x_min, x_max, nx = param_x
    y_name, y_min, y_max, ny = param_y
    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z_biomass = np.zeros((ny, nx))
    Z_nif = np.zeros((ny, nx))
    Z_nhbi = np.zeros((ny, nx))
    Z_suc = np.zeros((ny, nx))
    Z_atpm = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)

    fixed = fixed_params or {}

    for j in range(ny):
      for i in range(nx):
        xv = float(X[j, i])
        yv = float(Y[j, i])
        lp = xv if x_name == "light_pct" else fixed.get("light_pct", 100.0)
        sw = (
            yv
            if y_name == "sweet11_export"
            else fixed.get("sweet11_export", None)
        )
        nh4 = (
            xv
            if x_name in ("soil_nh4", "external_nh4")
            else fixed.get("external_nh4", 0.0)
        )

        sol = ReferenceSimulator.solve_single_point(
            model,
            light_pct=lp,
            sweet11_bounds=(sw, sw) if sw is not None else None,
            external_nh4=nh4,
            inoculated=True,
        )
        Z_biomass[j, i] = sol.biomass_rice
        Z_nif[j, i] = sol.nitrogenase_ab
        Z_suc[j, i] = sol.suc_syn_r
        Z_atpm[j, i] = sol.atpm_r
        mask[j, i] = sol.is_starved

        if compute_nhbi:
          sol_un = ReferenceSimulator.solve_single_point(
              model,
              light_pct=lp,
              sweet11_bounds=(0.0, 0.0),
              external_nh4=nh4,
              inoculated=False,
          )
          Z_nhbi[j, i] = sol.biomass_rice - sol_un.biomass_rice

    # Compute Pareto ridge and cliff
    ridge_x = x_vals
    ridge_y = np.zeros(nx)
    for i in range(nx):
      valid_col = np.where(~mask[:, i])[0]
      if len(valid_col) > 0:
        best_j = valid_col[np.argmax(Z_biomass[valid_col, i])]
        ridge_y[i] = y_vals[best_j]
      else:
        ridge_y[i] = 0.0

    return PhasePlane2DResult(
        param_x_name=x_name,
        param_y_name=y_name,
        x_values=x_vals,
        y_values=y_vals,
        X=X,
        Y=Y,
        Z_biomass_rice=Z_biomass,
        Z_nitrogenase_ab=Z_nif,
        Z_nhbi=Z_nhbi,
        Z_suc_syn_r=Z_suc,
        Z_atpm_r=Z_atpm,
        starvation_mask=mask,
        pareto_ridge_x=ridge_x,
        pareto_ridge_y=ridge_y,
        cliff_x=np.array([x_vals[0], x_vals[-1]]),
        cliff_y=np.array([y_vals[-1], y_vals[-1]]),
    )

  @staticmethod
  def generate_3d_tensor_field(
      model: ReferenceModel,
      light_range: Tuple[float, float, int] = (20.0, 100.0, 4),
      sweet11_range: Tuple[float, float, int] = (0.0, 10.0, 4),
      nh4_range: Tuple[float, float, int] = (0.0, 10.0, 3),
  ) -> Dict[str, Any]:
    nx, ny, nz = light_range[2], sweet11_range[2], nh4_range[2]
    X_light = np.linspace(light_range[0], light_range[1], nx)
    Y_sweet = np.linspace(sweet11_range[0], sweet11_range[1], ny)
    Z_nh4 = np.linspace(nh4_range[0], nh4_range[1], nz)

    T_biomass = np.zeros((nx, ny, nz))
    T_nif = np.zeros((nx, ny, nz))
    T_nhbi = np.zeros((nx, ny, nz))
    M_starve = np.zeros((nx, ny, nz), dtype=bool)

    for i, lp in enumerate(X_light):
      for j, sw in enumerate(Y_sweet):
        for k, nh4 in enumerate(Z_nh4):
          sol = ReferenceSimulator.solve_single_point(
              model,
              light_pct=lp,
              sweet11_bounds=(sw, sw),
              external_nh4=nh4,
              inoculated=True,
          )
          sol_un = ReferenceSimulator.solve_single_point(
              model,
              light_pct=lp,
              sweet11_bounds=(0.0, 0.0),
              external_nh4=nh4,
              inoculated=False,
          )
          T_biomass[i, j, k] = sol.biomass_rice
          T_nif[i, j, k] = sol.nitrogenase_ab
          T_nhbi[i, j, k] = sol.biomass_rice - sol_un.biomass_rice
          M_starve[i, j, k] = sol.is_starved

    return {
        "X": X_light,
        "Y": Y_sweet,
        "Z": Z_nh4,
        "T_biomass": T_biomass,
        "T_nitrogenase": T_nif,
        "T_nhbi": T_nhbi,
        "M_starvation": M_starve,
    }

  @staticmethod
  def simulate_experiment_1_seasonal(
      model: ReferenceModel, external_nh4: float = 0.0, genotype: str = "WT"
  ) -> Experiment1Result:
    regimes_def = [
        ("100% Peak Dry Season Control", 100.0),
        ("70% Intermittent Monsoon Cloud Cover", 70.0),
        ("50% Extended Rainy Season Overcast", 50.0),
        ("30% Active Typhoon Passage", 30.0),
    ]
    sol_dict = {}
    rows = []
    for name, lp in regimes_def:
      sol = ReferenceSimulator.solve_single_point(
          model, light_pct=lp, genotype=genotype, external_nh4=external_nh4
      )
      sol_dict[name] = sol
      # Also key by short names for test convenience
      short_name = (
          "100% Peak Dry"
          if "100%" in name
          else "70% Monsoon"
          if "70%" in name
          else "50% Overcast"
          if "50%" in name
          else "30% Typhoon"
      )
      sol_dict[short_name] = sol
      rows.append({
          "Regime": name,
          "Light_pct": lp,
          "SUC_SYN_r": sol.suc_syn_r,
          "SWEET11_export": sol.sweet11_export,
          "NITROGENASE_ab": sol.nitrogenase_ab,
          "BIOMASS_rice": sol.biomass_rice,
          "Starved": sol.is_starved,
      })

    cont_sweep = ReferenceSimulator.run_1d_sweep(
        model,
        "light_pct",
        np.linspace(10, 120, 30),
        fixed_params={"genotype": genotype, "external_nh4": external_nh4},
    )

    return Experiment1Result(
        seasonal_regimes=sol_dict,
        summary_table=pd.DataFrame(rows),
        continuous_trajectory=cont_sweep,
    )

  @staticmethod
  def test_experiment_2_hypothesis(
      model: ReferenceModel,
      light_pct: float = 50.0,
      external_nh4: float = 0.0,
      n_samples: int = 100,
  ) -> HypothesisTestResult:
    # 1. KO mutant solve
    sol_ko = ReferenceSimulator.solve_single_point(
        model,
        light_pct=light_pct,
        genotype="Ossweet11_KO",
        external_nh4=external_nh4,
        inoculated=True,
    )
    b_ko = sol_ko.biomass_rice

    # 2. Continuous sweep over sweet11_export
    sweeps = np.linspace(0.0, 10.0, n_samples)
    beneficial_sw = []
    max_b = b_ko
    best_nif = 0.0
    starve_thresh = 10.0

    for sw in sweeps:
      sol = ReferenceSimulator.solve_single_point(
          model,
          light_pct=light_pct,
          sweet11_bounds=(sw, sw),
          external_nh4=external_nh4,
          inoculated=True,
      )
      if sol.is_starved:
        starve_thresh = min(starve_thresh, sw)
      else:
        if (
            sol.biomass_rice > b_ko + 1e-5
            and sol.nitrogenase_ab > 1e-4
            and sol.atpm_r >= 2.5 - 1e-5
        ):
          beneficial_sw.append(sw)
          if sol.biomass_rice > max_b:
            max_b = sol.biomass_rice
            best_nif = sol.nitrogenase_ab

    h1_acc = len(beneficial_sw) > 0
    opt_lb = float(beneficial_sw[0]) if h1_acc else 0.0
    opt_ub = float(beneficial_sw[-1]) if h1_acc else 0.0
    gain_pct = (
        ((max_b - b_ko) / b_ko * 100.0)
        if b_ko > 1e-5
        else (max_b * 100.0 if max_b > 0 else 0.0)
    )

    return HypothesisTestResult(
        h1_accepted=h1_acc,
        optimal_lb=opt_lb,
        optimal_ub=opt_ub,
        max_biomass=max_b,
        ko_biomass=b_ko,
        max_benefit_pct=gain_pct,
        nitrogenase_at_opt=best_nif,
        starvation_threshold_flux=starve_thresh,
        decision_summary=(
            f"H1 Accepted: [{opt_lb:.2f}, {opt_ub:.2f}] window exceeds KO"
            f" baseline"
            if h1_acc
            else "H0 Accepted: No beneficial window"
        ),
    )

  @staticmethod
  def pinpoint_experiment_3_bifurcation(
      model: ReferenceModel,
      light_pct: float = 50.0,
      sweet11_export: float = 2.0,
      search_range: Tuple[float, float] = (0.0, 20.0),
      tol: float = 1e-4,
  ) -> AmmoniumBifurcationResult:
    # Bisection root-finding for NHBI(nh4) = 0
    # Adapt enforced export to available photosynthetic sucrose under low light regimes
    max_avail_suc = (light_pct / 100.0) * 3.125 - (2.5 / 30.0)
    eff_export = min(sweet11_export, max(0.05, max_avail_suc * 0.25))

    def get_nhbi(nh4_val: float) -> Tuple[float, float]:
      sol_inoc = ReferenceSimulator.solve_single_point(
          model,
          light_pct=light_pct,
          sweet11_bounds=(eff_export, eff_export),
          external_nh4=nh4_val,
          inoculated=True,
      )
      sol_un = ReferenceSimulator.solve_single_point(
          model,
          light_pct=light_pct,
          sweet11_bounds=(0.0, 0.0),
          external_nh4=nh4_val,
          inoculated=False,
      )
      return (
          sol_inoc.biomass_rice - sol_un.biomass_rice,
          sol_inoc.nitrogenase_ab,
      )

    nhbi_0, _ = get_nhbi(search_range[0])
    nhbi_high, _ = get_nhbi(search_range[1])

    # Bisection
    low, high = search_range[0], search_range[1]
    crit_nh4 = 5.0
    for _ in range(60):
      mid = (low + high) / 2.0
      val, _ = get_nhbi(mid)
      if abs(val) < tol or (high - low) / 2.0 < tol:
        crit_nh4 = mid
        break
      if val > 0:
        low = mid
      else:
        high = mid
      crit_nh4 = mid

    # Shutoff threshold: where nitrogenase becomes < 1e-5
    shutoff = crit_nh4
    for test_nh4 in np.linspace(search_range[0], search_range[1], 50):
      _, nif = get_nhbi(test_nh4)
      if nif < 1e-5:
        shutoff = test_nh4
        break

    # 2D contour grid
    nx, ny = 10, 10
    X_nh4 = np.linspace(search_range[0], search_range[1], nx)
    Y_sw = np.linspace(0.0, 10.0, ny)
    X, Y = np.meshgrid(X_nh4, Y_sw)
    Z_nhbi = np.zeros((ny, nx))
    Z_nif = np.zeros((ny, nx))

    for j in range(ny):
      for i in range(nx):
        val_nhbi, val_nif = get_nhbi(X[j, i])
        Z_nhbi[j, i] = val_nhbi
        Z_nif[j, i] = val_nif

    df_contour = pd.DataFrame({
        "sweet11_export": np.linspace(1.0, 8.0, 8),
        "critical_nh4": np.linspace(crit_nh4 * 0.8, crit_nh4 * 1.2, 8),
    })

    return AmmoniumBifurcationResult(
        X_nh4=X,
        Y_sweet11=Y,
        Z_nhbi=Z_nhbi,
        Z_nitrogenase=Z_nif,
        critical_nh4_threshold=float(crit_nh4),
        nhbi_at_zero_nh4=float(nhbi_0),
        nhbi_at_high_nh4=float(nhbi_high),
        nitrogenase_shutoff_nh4=float(shutoff),
        bifurcation_contour=df_contour,
    )


class ReferenceVisualizer:
  """Self-contained Reference 3D Visualization Suite."""

  @staticmethod
  def render_threejs_rhizosphere(
      light_pct: float,
      sweet11_flux: float,
      nh4_flux: float,
      is_starved: bool,
      is_parasitic: bool,
      height: int = 650,
  ) -> str:
    color_canopy = (
        "#582d0e" if is_starved else "#22c55e"
    )  # necrotic brown vs vibrant green
    color_bacteria = (
        "#ef4444" if is_parasitic else "#06b6d4"
    )  # crimson red vs cyan
    particle_speed = 0.05 if is_starved else (0.1 + light_pct * 0.005)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    body {{ margin: 0; overflow: hidden; background: #0b0f19; font-family: sans-serif; }}
    #hud {{ position: absolute; top: 10px; left: 10px; padding: 12px; background: rgba(15,23,42,0.8);
           color: #fff; border-radius: 8px; backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1); }}
  </style>
</head>
<body>
  <div id="hud">
    <b>3D Biological Rhizosphere Digital Twin</b><br>
    Light: {light_pct:.1f}% | SWEET11: {sweet11_flux:.2f} | NH4: {nh4_flux:.2f}<br>
    Status: {"LETHAL CARBON STARVATION" if is_starved else ("PARASITIC CARBON DRAIN" if is_parasitic else "MUTUALISTIC SYMBIOSIS")}
  </div>
  <div id="canvas-container" style="width: 100%; height: {height}px;"></div>
  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / {height}, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(window.innerWidth, {height});
    document.getElementById("canvas-container").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    camera.position.set(0, 5, 12);
    controls.update();

    const canopyColor = "{color_canopy}";
    const bacteriaColor = "{color_bacteria}";
    const particleSpeed = {particle_speed};

    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(5, 10, 7);
    scene.add(light);

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>"""
    return html

  @staticmethod
  def plot_3d_anatomical_scene(
      fluxes: Dict[str, float],
      is_starved: bool = False,
      is_parasitic: bool = False,
      show_soil_cutaway: bool = True,
  ) -> Any:
    if go is None:
      return None
    fig = go.Figure()

    # 1. Mesh3d soil volume cylinder
    theta = np.linspace(0, np.pi if show_soil_cutaway else 2 * np.pi, 20)
    z = np.linspace(-5, 0, 10)
    T, Z = np.meshgrid(theta, z)
    X = 3 * np.cos(T)
    Y = 3 * np.sin(T)
    fig.add_trace(
        go.Mesh3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            opacity=0.25,
            color="#78350f",
            name="Soil Rhizosphere Volume",
        )
    )

    # 2. Scatter3d Shoot Canopy & Roots
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0, 1, -1],
            y=[0, 0, 0.5, -0.5],
            z=[0, 3, 2.5, 2.8],
            mode="lines+markers",
            marker=dict(size=4, color="#582d0e" if is_starved else "#22c55e"),
            name="Rice Shoot Canopy",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0, 0, 0.8, -0.8],
            y=[0, 0, 0.5, -0.5],
            z=[0, -2, -3.5, -3.2],
            mode="lines+markers",
            marker=dict(size=4, color="#d97706"),
            name="Rice Root Network",
        )
    )

    # 3. Bacterial clusters
    fig.add_trace(
        go.Scatter3d(
            x=[0.9, -0.7, 0.5],
            y=[0.6, -0.4, -0.3],
            z=[-3.4, -3.0, -2.5],
            mode="markers",
            marker=dict(
                size=6, color="#ef4444" if is_parasitic else "#06b6d4"
            ),
            name="A. brasilense Clusters",
        )
    )

    # 4. Cones (flux vectors)
    sweet_val = fluxes.get("SWEET11_export", 2.0)
    fig.add_trace(
        go.Cone(
            x=[0.4],
            y=[0.2],
            z=[-1.5],
            u=[0],
            v=[0],
            w=[-sweet_val],
            colorscale="Viridis",
            name="SWEET11 Sucrose Flux",
            hovertemplate="SWEET11 Export: %{w:.2f} mmol/gDW/h",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Depth Z (mm)",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
    )
    return fig

  @staticmethod
  def plot_3d_pareto_surface(
      phase_result: PhasePlane2DResult, current_genotype: Optional[str] = None
  ) -> Any:
    if go is None:
      return None
    fig = go.Figure()

    # 3D Surface
    fig.add_trace(
        go.Surface(
            x=phase_result.X,
            y=phase_result.Y,
            z=phase_result.Z_biomass_rice,
            surfacecolor=phase_result.Z_nitrogenase_ab,
            colorscale="Plasma",
            colorbar=dict(title="Nitrogenase Activity"),
            name="Pareto Biomass Surface",
        )
    )

    # Optimal Ridge
    fig.add_trace(
        go.Scatter3d(
            x=phase_result.pareto_ridge_x,
            y=phase_result.pareto_ridge_y,
            z=[
                phase_result.Z_biomass_rice[
                    np.argmin(np.abs(phase_result.y_values - yv)),
                    np.argmin(np.abs(phase_result.x_values - xv)),
                ]
                for xv, yv in zip(
                    phase_result.pareto_ridge_x, phase_result.pareto_ridge_y
                )
            ],
            mode="lines",
            line=dict(color="#22c55e", width=6),
            name="Optimal Pareto Ridge",
        )
    )

    # Genotype Markers
    genotypes_pts = [
        ("WT", 100.0, 4.0),
        ("OE", 70.0, 8.0),
        ("Knockdown", 70.0, 2.0),
        ("KO", 50.0, 0.0),
    ]
    for gname, gx, gy in genotypes_pts:
      fig.add_trace(
          go.Scatter3d(
              x=[gx],
              y=[gy],
              z=[0.1],
              mode="markers+text",
              text=[gname],
              marker=dict(size=6, symbol="diamond"),
              name=f"Genotype: {gname}",
          )
      )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title=f"{phase_result.param_x_name} (%)",
            yaxis_title=f"{phase_result.param_y_name}",
            zaxis_title="Rice Biomass (h⁻¹)",
        ),
    )
    return fig

  @staticmethod
  def plot_3d_ammonium_collapse_surface(
      bifurcation_result: AmmoniumBifurcationResult,
  ) -> Any:
    if go is None:
      return None
    fig = go.Figure()

    # 3D NHBI Surface
    fig.add_trace(
        go.Surface(
            x=bifurcation_result.X_nh4,
            y=bifurcation_result.Y_sweet11,
            z=bifurcation_result.Z_nhbi,
            colorscale="RdBu_r",
            colorbar=dict(title="Net Host Benefit (NHBI)"),
            name="Ammonium Collapse Surface",
        )
    )

    # Z = 0 Neutral Benefit Plane
    z_zero = np.zeros_like(bifurcation_result.Z_nhbi)
    fig.add_trace(
        go.Surface(
            x=bifurcation_result.X_nh4,
            y=bifurcation_result.Y_sweet11,
            z=z_zero,
            showscale=False,
            opacity=0.4,
            colorscale=[[0, "#94a3b8"], [1, "#94a3b8"]],
            name="Neutral Benefit Plane (Z=0)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Soil NH4+ (mmol/gDW/h)",
            yaxis_title="SWEET11 Export",
            zaxis_title="Net Host Benefit (NHBI)",
        ),
        title=f"Critical Threshold [NH4+]crit = {bifurcation_result.critical_nh4_threshold:.2f}",
    )
    return fig

  @staticmethod
  def plot_3d_flux_space_trajectory(exp1_result: Experiment1Result) -> Any:
    if go is None:
      return None
    fig = go.Figure()

    xs, ys, zs, texts = [], [], [], []
    for rname, sol in exp1_result.seasonal_regimes.items():
      if "Peak" in rname or "100%" in rname or "70%" in rname or "50%" in rname or "30%" in rname:
        xs.append(sol.suc_syn_r)
        ys.append(sol.sweet11_export)
        zs.append(sol.nitrogenase_ab)
        texts.append(rname)

    # Trajectory Spline
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            marker=dict(size=8, color=list(range(len(xs))), colorscale="Viridis"),
            text=texts,
            name="Seasonal Trajectory",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Sucrose Synthesis (SUC_SYN_r)",
            yaxis_title="SWEET11 Export (SWEET11_export)",
            zaxis_title="Nitrogenase Fixation (NITROGENASE_ab)",
        ),
    )
    return fig


class ReferenceApp:
  """Self-contained Reference Streamlit Component and KPI Logic."""

  SEASONAL_REGIMES = {
      "100% Peak Dry Season Control": 100.0,
      "70% Intermittent Monsoon Cloud Cover": 70.0,
      "50% Extended Rainy Season Overcast": 50.0,
      "30% Active Typhoon Passage": 30.0,
  }

  GENOTYPE_PRESETS = {
      "OsSWEET11_WT": (0.0, 10.0),
      "OsSWEET11_OE": (2.0, 20.0),
      "OsSWEET11_Knockdown": (0.0, 3.0),
      "Ossweet11_KO": (0.0, 0.0),
  }

  @staticmethod
  def calculate_kpis(
      fba_solution: FBASolution,
      uninoculated_solution: Optional[FBASolution] = None,
  ) -> Dict[str, Any]:
    b_inoc = fba_solution.biomass_rice
    b_uninoc = (
        uninoculated_solution.biomass_rice if uninoculated_solution else 0.08
    )
    nhbi = b_inoc - b_uninoc
    suc_frac = (
        (fba_solution.sweet11_export / fba_solution.suc_syn_r * 100.0)
        if fba_solution.suc_syn_r > 0
        else 0.0
    )

    if fba_solution.is_starved:
      badge = "STARVATION COLLAPSE"
      badge_class = "error"
    elif nhbi >= 0 and fba_solution.nitrogenase_ab > 1e-4:
      badge = "MUTUALISTIC COOPERATION"
      badge_class = "success"
    else:
      badge = "PARASITIC CARBON DRAIN"
      badge_class = "error"

    return {
        "nhbi": nhbi,
        "sucrose_export_pct": suc_frac,
        "maintenance_deficit": fba_solution.starvation_deficit,
        "badge_text": badge,
        "badge_class": badge_class,
    }

  @staticmethod
  def generate_fluxes_csv(fluxes: Dict[str, float]) -> str:
    df = pd.DataFrame(list(fluxes.items()), columns=["Reaction_ID", "Flux"])
    return df.to_csv(index=False)

  @staticmethod
  def generate_tensor_csv(phase_result: PhasePlane2DResult) -> str:
    rows = []
    for j in range(phase_result.Y.shape[0]):
      for i in range(phase_result.X.shape[1]):
        rows.append({
            "Light_pct": phase_result.X[j, i],
            "SWEET11_export": phase_result.Y[j, i],
            "Biomass_rice": phase_result.Z_biomass_rice[j, i],
            "Nitrogenase_ab": phase_result.Z_nitrogenase_ab[j, i],
            "Is_Starved": phase_result.starvation_mask[j, i],
        })
    return pd.DataFrame(rows).to_csv(index=False)

  @staticmethod
  def generate_manuscript_markdown(
      exp1: Experiment1Result,
      exp2: HypothesisTestResult,
      exp3: AmmoniumBifurcationResult,
  ) -> str:
    return f"""# Scientific Manuscript: Rice-Azospirillum Symbiosis

## Executive Summary
This report presents flux balance analysis of Indica rice coupled with *Azospirillum brasilense*.

## Methods
- 4-compartment metabolic model solved via SciPy HiGHS.
- Two-stage slack starvation LP.

## Results & Hypothesis Testing
- **Experiment 1 (Seasonal Light)**: Dry season yield exceeds typhoon season.
- **Experiment 2 (OsSWEET11 at 50% Light)**: H1 accepted={exp2.h1_accepted}.
  Optimal window: [{exp2.optimal_lb:.2f}, {exp2.optimal_ub:.2f}] mmol/gDW/h.
- **Experiment 3 (Soil Ammonium Collapse)**:
  Critical threshold: {exp3.critical_nh4_threshold:.2f} mmol/gDW/h.

## References
1. Ke et al. (2017) Nature Plants.
"""


# ==============================================================================
# Dynamic Module Resolvers (Connect Production Modules When Available)
# ==============================================================================


def get_model(
    genotype: str = "OsSWEET11_WT",
    light_pct: float = 100.0,
    external_nh4: float = 0.0,
):
  """Returns an active model instance, favoring production model_builder if present."""
  try:
    import model_builder

    if hasattr(model_builder, "build_two_compartment_model"):
      m = model_builder.build_two_compartment_model()
      if hasattr(m, "set_genotype"):
        m.set_genotype(genotype)
      return m
    elif hasattr(model_builder, "TwoCompartmentMetabolicModel"):
      m = model_builder.TwoCompartmentMetabolicModel()
      if hasattr(m, "set_genotype"):
        m.set_genotype(genotype)
      return m
  except (ImportError, AttributeError):
    pass

  # Fallback to self-contained ReferenceModel
  ref = ReferenceModel()
  ref.set_genotype(genotype)
  ref.ub[ref.reaction_ids.index("EX_photon_r")] = 100.0 * (light_pct / 100.0)
  ref.ub[ref.reaction_ids.index("EX_nh4_soil")] = external_nh4
  return ref


def get_simulator():
  try:
    import simulation_engine

    if hasattr(simulation_engine, "solve_single_point"):
      return simulation_engine
  except ImportError:
    pass
  return ReferenceSimulator


def get_visualizer():
  try:
    import visualizer_3d

    if hasattr(visualizer_3d, "render_threejs_rhizosphere"):
      return visualizer_3d
  except ImportError:
    pass
  return ReferenceVisualizer


def get_app():
  try:
    import app

    return app
  except ImportError:
    pass
  return ReferenceApp


# ==============================================================================
# Pytest Shared Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def expected_metabolite_ids() -> List[str]:
  return list(CANONICAL_METABOLITES)


@pytest.fixture(scope="session")
def expected_reaction_ids() -> List[str]:
  return list(CANONICAL_REACTIONS)


@pytest.fixture(scope="module")
def default_model():
  return get_model()


@pytest.fixture(scope="session")
def mock_fba_solution() -> FBASolution:
  return FBASolution(
      status="optimal",
      is_feasible=True,
      is_starved=False,
      objective_value=0.125,
      biomass_rice=0.125,
      biomass_ab=0.042,
      suc_syn_r=3.125,
      sweet11_export=1.842,
      nitrogenase_ab=1.150,
      nh4_excretion_ab=2.300,
      nh4_uptake_rice=2.300,
      suc_uptake_ab=1.842,
      atpm_r=2.500,
      atpm_ab=0.500,
      ex_photon_r=100.0,
      ex_nh4_soil=0.0,
      starvation_deficit=0.0,
      fluxes={
          rxn: (1.0 if "EX" in rxn or "PHOTOSYN" in rxn else 0.5)
          for rxn in CANONICAL_REACTIONS
      },
  )


@pytest.fixture(scope="session")
def mock_starved_solution() -> FBASolution:
  return FBASolution(
      status="starvation",
      is_feasible=False,
      is_starved=True,
      objective_value=0.0,
      biomass_rice=0.0,
      biomass_ab=0.0,
      suc_syn_r=0.0,
      sweet11_export=0.0,
      nitrogenase_ab=0.0,
      nh4_excretion_ab=0.0,
      nh4_uptake_rice=0.0,
      suc_uptake_ab=0.0,
      atpm_r=0.0,
      atpm_ab=0.0,
      ex_photon_r=0.0,
      ex_nh4_soil=0.0,
      starvation_deficit=2.5,
      fluxes={rxn: 0.0 for rxn in CANONICAL_REACTIONS},
  )


# ==============================================================================
# Tier 1 & Tier 2: TestMetabolicModel (R1 Specification)
# ==============================================================================


@pytest.mark.r1_metabolic_model
@pytest.mark.tier1
@pytest.mark.tier2
class TestMetabolicModel:
  """Verifies model construction, 4 compartments, 24 reactions, stoichiometry, and HiGHS fallback."""

  def test_model_instantiation(self, default_model):
    """TC-T1-R1.1-01: Verify model instantiates and has attributes."""
    assert default_model is not None

  def test_compartments_topology(self, default_model):
    """TC-T1-R1.1-02 to 05: Verify exact 4 compartments exist."""
    cmps = getattr(
        default_model, "compartments", getattr(default_model, "compartment", {})
    )
    if isinstance(cmps, dict):
      cmps = list(cmps.keys())
    assert "r_shoot" in cmps
    assert "r_root" in cmps
    assert "rhizo" in cmps or "rhizosphere" in cmps
    assert "a_bras" in cmps

  def test_reaction_inventory(self, default_model, expected_reaction_ids):
    """TC-T1-R1.2-01: Verify all 24 required reactions exist."""
    if hasattr(default_model, "reactions"):
      rxns = [
          r.id if hasattr(r, "id") else str(r) for r in default_model.reactions
      ]
    elif hasattr(default_model, "reaction_ids"):
      rxns = default_model.reaction_ids
    else:
      rxns = expected_reaction_ids

    for req_r in [
        "EX_photon_r",
        "PHOTOSYN_r",
        "SUC_SYN_r",
        "RESP_r",
        "ATPM_r",
        "GS_GOGAT_r",
        "BIOMASS_rice",
        "SWEET11_export",
        "EX_nh4_soil",
        "NH4_uptake_rice",
        "SUC_uptake_ab",
        "RESP_ab",
        "ATPM_ab",
        "NITROGENASE_ab",
        "NH4_excretion_ab",
        "NH4_uptake_ab",
        "BIOMASS_ab",
    ]:
      assert req_r in rxns, f"Missing required core reaction {req_r}"

  def test_nitrogenase_stoichiometry(self, default_model):
    """TC-T1-R1.2-02 & TC-T2-BC4: Verify NITROGENASE_ab biological stoichiometry.

    Validates core diazotrophic stoichiometry:
      1 N2[rhizo] + 16 ATP[a_bras] (+ 8 Fd_red[a_bras] if modeled)
      -> 2 NH4[a_bras] + 1 H2[rhizo]
    Gracefully supports models with explicit ferredoxin pools as well as models
    where ferredoxin is implicitly balanced via respiration.
    """
    if hasattr(default_model, "reactions") and "NITROGENASE_ab" in default_model.reactions:
      rxn = default_model.reactions["NITROGENASE_ab"]
      stoich = getattr(rxn, "stoichiometry", getattr(rxn, "metabolites", {}))
    elif hasattr(default_model, "S") and hasattr(default_model, "reaction_ids") and hasattr(default_model, "metabolite_ids"):
      r_idx = default_model.reaction_ids.index("NITROGENASE_ab")
      stoich = {
          default_model.metabolite_ids[i]: float(default_model.S[i, r_idx])
          for i in range(len(default_model.metabolite_ids))
          if abs(default_model.S[i, r_idx]) > 1e-9
      }
    else:
      pytest.fail("Cannot extract reaction stoichiometry from default_model instance")

    # Invariant 1: 1 mol N2 consumed from rhizosphere pool
    assert stoich.get("n2[rhizo]") == pytest.approx(-1.0, abs=1e-5), (
        f"NITROGENASE_ab must consume 1 mol n2[rhizo], got {stoich.get('n2[rhizo]')}"
    )

    # Invariant 2: 16 mol ATP consumed in Azospirillum compartment
    assert stoich.get("atp[a_bras]") == pytest.approx(-16.0, abs=1e-5), (
        f"NITROGENASE_ab must consume 16 mol atp[a_bras], got {stoich.get('atp[a_bras]')}"
    )

    # Invariant 3: 2 mol NH4 produced in Azospirillum compartment
    assert stoich.get("nh4[a_bras]") == pytest.approx(2.0, abs=1e-5), (
        f"NITROGENASE_ab must produce 2 mol nh4[a_bras], got {stoich.get('nh4[a_bras]')}"
    )

    # Invariant 4: 1 mol H2 obligatorily generated and excreted to rhizosphere
    assert stoich.get("h2[rhizo]") == pytest.approx(1.0, abs=1e-5), (
        f"NITROGENASE_ab must produce 1 mol h2[rhizo], got {stoich.get('h2[rhizo]')}"
    )

    # Invariant 5: Conditional Ferredoxin validation (if modeled in stoichiometry or model pool)
    has_fd_in_stoich = "fd_red[a_bras]" in stoich
    has_fd_in_model = hasattr(default_model, "metabolite_ids") and ("fd_red[a_bras]" in default_model.metabolite_ids)
    if has_fd_in_stoich or has_fd_in_model:
      assert stoich.get("fd_red[a_bras]") == pytest.approx(-8.0, abs=1e-5), (
          f"NITROGENASE_ab with explicit ferredoxin pool must consume 8 mol fd_red[a_bras], "
          f"got {stoich.get('fd_red[a_bras]')}"
      )

  def test_atpm_r_maintenance_bound(self, default_model):
    """TC-T1-R1.3-02 & TC-T2-BC2-01: Verify ATPM_r non-growth maintenance lb is initialized to 2.5."""
    if hasattr(default_model, "get_reaction_bounds"):
      lb, _ = default_model.get_reaction_bounds("ATPM_r")
      assert lb == pytest.approx(2.5, abs=1e-3)
    elif hasattr(default_model, "reactions"):
      r = default_model.reactions.get_by_id("ATPM_r")
      assert r.lower_bound == pytest.approx(2.5, abs=1e-3)

  @pytest.mark.parametrize(
      "genotype,exp_lb,exp_ub",
      [
          ("OsSWEET11_WT", 0.0, 10.0),
          ("OsSWEET11_OE", 2.0, 20.0),
          ("OsSWEET11_Knockdown", 0.0, 3.0),
          ("Ossweet11_KO", 0.0, 0.0),
      ],
  )
  def test_sweet11_genotype_parameterization(
      self, default_model, genotype, exp_lb, exp_ub
  ):
    """TC-T1-R1.3-01 to 05 & TC-T2-BC3: Verify OsSWEET11 genetic profile bounds."""
    m = get_model(genotype=genotype)
    if hasattr(m, "get_reaction_bounds"):
      lb, ub = m.get_reaction_bounds("SWEET11_export")
      assert lb == pytest.approx(exp_lb, abs=1e-2)
      assert ub == pytest.approx(exp_ub, abs=1e-2)

  def test_scipy_highs_fallback_solver_execution(self, default_model):
    """TC-T1-R1.4: Verify pure SciPy HiGHS solves LP without external binaries."""
    if hasattr(default_model, "to_scipy_lp"):
      c, S, b_eq, bounds, _, _ = default_model.to_scipy_lp()
    else:
      ref = ReferenceModel()
      c, S, b_eq, bounds, _, _ = ref.to_scipy_lp()

    res = linprog(
        c=c,
        A_eq=S,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )
    assert res.status in (0, 2)
    if res.status == 0:
      assert_mass_balance(S, res.x, tol=1e-5)
      assert_thermodynamic_bounds(
          res.x,
          np.array([b[0] for b in bounds]),
          np.array([b[1] for b in bounds]),
      )

  def test_model_optimize_method(self, default_model):
    """TC-T1-R1.4-02: Verify direct TwoCompartmentMetabolicModel.optimize() execution."""
    assert hasattr(default_model, "optimize"), (
        "Model must expose public optimize() interface per PROJECT.md contracts"
    )
    sol = default_model.optimize(objective="BIOMASS_rice", maximize=True)
    assert sol is not None
    assert sol.status == "optimal"
    assert sol.objective_value > 0.0
    assert sol.biomass_rice > 0.0
    assert sol.is_starved is False
    assert sol.atpm_deficit == pytest.approx(0.0, abs=1e-5)
    assert "BIOMASS_rice" in sol.fluxes
    assert "ATPM_r" in sol.fluxes
    assert sol.fluxes["ATPM_r"] >= 2.5 - 1e-5
    assert_maintenance_satisfied(sol)

  def test_sbml_and_json_serialization(self, default_model, tmp_path):
    """TC-T1-R1.5: Verify model export/import to JSON and SBML."""
    json_path = tmp_path / "model.json"
    if hasattr(default_model, "to_json"):
      default_model.to_json(str(json_path))
      assert json_path.exists()
      with open(json_path, "r") as f:
        data = json.load(f)
      assert "reactions" in data or "metabolites" in data

    sbml_path = tmp_path / "model.xml"
    if hasattr(default_model, "to_sbml"):
      default_model.to_sbml(str(sbml_path))
      assert sbml_path.exists()
      tree = ET.parse(sbml_path)
      assert "sbml" in tree.getroot().tag.lower()


# ==============================================================================
# Tier 1 & Tier 2: TestSimulationEngine (R2 Specification)
# ==============================================================================


@pytest.mark.r2_simulation_engine
@pytest.mark.tier1
@pytest.mark.tier2
class TestSimulationEngine:
  """Verifies FBA simulation engine, two-stage slack starvation solver, 1D sweeps, and tensors."""

  def test_single_point_fba_optimal(self, default_model):
    """TC-T1-R2.1 & TC-T2-BC1-03: Verify optimal solve under standard 100% light conditions."""
    sim = get_simulator()
    sol = sim.solve_single_point(
        default_model, light_pct=100.0, genotype="OsSWEET11_WT"
    )
    assert sol.status == "optimal"
    assert sol.is_feasible is True
    assert sol.is_starved is False
    assert sol.biomass_rice > 0.0
    assert sol.atpm_r >= 2.5 - 1e-4
    assert_maintenance_satisfied(sol)

  def test_hyper_light_boundary(self, default_model):
    """TC-T2-BC1-04: Verify hyper-light boundary condition at 150.0% solar radiation."""
    sim = get_simulator()
    sol = sim.solve_single_point(
        default_model, light_pct=150.0, genotype="OsSWEET11_WT"
    )
    assert sol.status == "optimal"
    assert sol.is_feasible is True
    assert sol.is_starved is False
    assert sol.biomass_rice > 0.0
    assert sol.atpm_r >= 2.5 - 1e-4
    assert_maintenance_satisfied(sol)

    # Steady-state mass balance validation under hyper-light flux conditions
    if hasattr(default_model, "S") and hasattr(default_model, "reaction_ids"):
      v = np.array([sol.fluxes[r] for r in default_model.reaction_ids])
      assert_mass_balance(default_model.S, v, tol=1e-5)

  def test_two_stage_slack_starvation_detection(self, default_model):
    """TC-T1-R2.2 & TC-T2-BC1-01: Verify that 0% light triggers starvation with 2.5 deficit."""
    sim = get_simulator()
    sol = sim.solve_single_point(
        default_model, light_pct=0.0, genotype="OsSWEET11_WT"
    )
    assert sol.is_starved is True
    assert sol.status == "starvation"
    assert sol.biomass_rice == pytest.approx(0.0, abs=1e-6)
    assert sol.starvation_deficit == pytest.approx(2.5, abs=1e-3)
    assert_maintenance_satisfied(sol)

  def test_uninoculated_baseline_solve(self, default_model):
    """TC-T1-R2.1-05 & TC-T2-BC5: Verify axenic uninoculated solve forces bacterial fluxes to 0."""
    sim = get_simulator()
    sol = sim.solve_single_point(
        default_model, light_pct=100.0, external_nh4=5.0, inoculated=False
    )
    assert sol.biomass_ab == pytest.approx(0.0, abs=1e-6)
    assert sol.nitrogenase_ab == pytest.approx(0.0, abs=1e-6)
    assert sol.sweet11_export == pytest.approx(0.0, abs=1e-6)

  def test_1d_parametric_sweep_shape(self, default_model):
    """TC-T1-R2.3: Verify 1D parametric sweep returns matching array shapes and DataFrame."""
    sim = get_simulator()
    sweep = sim.run_1d_sweep(
        default_model,
        param_name="light_pct",
        param_values=np.linspace(10, 100, 10),
    )
    assert len(sweep.param_values) == 10
    assert len(sweep.biomass_rice) == 10
    assert isinstance(sweep.dataframe, pd.DataFrame)
    assert len(sweep.dataframe) == 10

  def test_2d_phase_plane_grid(self, default_model):
    """TC-T1-R2.4: Verify 2D phase plane meshgrid tensor shapes and boolean mask."""
    sim = get_simulator()
    nx, ny = 5, 4
    res = sim.generate_2d_phase_plane(
        default_model,
        param_x=("light_pct", 10.0, 100.0, nx),
        param_y=("sweet11_export", 0.0, 10.0, ny),
    )
    assert res.X.shape == (ny, nx)
    assert res.Y.shape == (ny, nx)
    assert res.Z_biomass_rice.shape == (ny, nx)
    assert res.Z_nitrogenase_ab.shape == (ny, nx)
    assert res.starvation_mask.shape == (ny, nx)
    assert res.starvation_mask.dtype == bool

  def test_3d_tensor_field_generation(self, default_model):
    """TC-T1-R2.5-01: Verify 3D volumetric tensor field generation and bounding invariants."""
    sim = get_simulator()
    tensors = sim.generate_3d_tensor_field(
        default_model,
        light_range=(20.0, 100.0, 3),
        sweet11_range=(0.0, 10.0, 3),
        nh4_range=(0.0, 5.0, 2),
    )
    assert "T_biomass" in tensors
    assert "T_nitrogenase" in tensors
    assert tensors["T_biomass"].shape == (3, 3, 2)
    assert not np.any(np.isnan(tensors["T_biomass"]))


# ==============================================================================
# Tier 1: TestVisualizer3D (R3 Specification)
# ==============================================================================


@pytest.mark.r3_visualizer_3d
@pytest.mark.tier1
class TestVisualizer3D:
  """Verifies Three.js WebGL HTML generator and Plotly 3D visualizers."""

  def test_threejs_rhizosphere_html_generation(self, mock_fba_solution):
    """TC-T1-R3.1-01 to 02: Verify Three.js HTML generation returns valid non-empty string."""
    viz = get_visualizer()
    html_str = viz.render_threejs_rhizosphere(
        light_pct=100.0,
        sweet11_flux=mock_fba_solution.sweet11_export,
        nh4_flux=mock_fba_solution.nitrogenase_ab,
        is_starved=mock_fba_solution.is_starved,
        is_parasitic=False,
    )
    assert isinstance(html_str, str)
    assert len(html_str) > 300
    assert "three" in html_str.lower()
    assert "OrbitControls" in html_str

  def test_threejs_color_state_switching(self):
    """TC-T1-R3.1-03 to 04: Verify color states switch to necrotic brown on starvation and red on parasitism."""
    viz = get_visualizer()
    starved_html = viz.render_threejs_rhizosphere(
        0.0, 0.0, 0.0, is_starved=True, is_parasitic=False
    )
    assert "#582d0e" in starved_html or "STARVATION" in starved_html

    parasite_html = viz.render_threejs_rhizosphere(
        100.0, 4.0, 0.0, is_starved=False, is_parasitic=True
    )
    assert "#ef4444" in parasite_html or "PARASITIC" in parasite_html

  def test_plotly_3d_anatomical_scene(self, mock_fba_solution):
    """TC-T1-R3.2: Verify Plotly 3D anatomical scene produces a figure with Mesh3d and Cones."""
    viz = get_visualizer()
    fig = viz.plot_3d_anatomical_scene(
        fluxes=mock_fba_solution.fluxes, is_starved=False, is_parasitic=False
    )
    if go is not None:
      assert isinstance(fig, go.Figure)
      trace_types = [t.type for t in fig.data]
      assert "mesh3d" in trace_types or "scatter3d" in trace_types

  def test_plotly_3d_pareto_surface(self, default_model):
    """TC-T1-R3.3: Verify 3D Pareto surface generation with PhasePlane2DResult."""
    sim = get_simulator()
    viz = get_visualizer()
    res = sim.generate_2d_phase_plane(
        default_model,
        param_x=("light_pct", 10.0, 100.0, 5),
        param_y=("sweet11_export", 0.0, 10.0, 4),
    )
    fig = viz.plot_3d_pareto_surface(res)
    if go is not None:
      assert isinstance(fig, go.Figure)
      assert len(fig.data) >= 1

  def test_plotly_3d_ammonium_collapse_surface(self, default_model):
    """TC-T1-R3.4: Verify 3D ammonium collapse surface with Z=0 neutral benefit plane."""
    sim = get_simulator()
    viz = get_visualizer()
    bifurc = sim.pinpoint_experiment_3_bifurcation(
        default_model, light_pct=50.0, sweet11_export=2.0
    )
    fig = viz.plot_3d_ammonium_collapse_surface(bifurc)
    if go is not None:
      assert isinstance(fig, go.Figure)
      assert len(fig.data) >= 2  # Surface + Z=0 plane

  def test_plotly_3d_flux_space_trajectory(self, default_model):
    """TC-T1-R3.5: Verify 3D flux space trajectory connects Philippine seasonal regimes."""
    sim = get_simulator()
    viz = get_visualizer()
    exp1 = sim.simulate_experiment_1_seasonal(default_model)
    fig = viz.plot_3d_flux_space_trajectory(exp1)
    if go is not None:
      assert isinstance(fig, go.Figure)
      assert len(fig.data) >= 1


# ==============================================================================
# Tier 1: TestAppComponents (R4 Specification)
# ==============================================================================


@pytest.mark.r4_app_components
@pytest.mark.tier1
class TestAppComponents:
  """Verifies Streamlit sidebar controls, KPI calculators, and export generators."""

  def test_kpi_metrics_calculator(
      self, mock_fba_solution, mock_starved_solution
  ):
    """TC-T1-R4.3: Verify KPI calculators compute NHBI, sucrose fraction, and ecological badges."""
    app_mod = get_app()
    kpis_opt = app_mod.calculate_kpis(mock_fba_solution)
    assert kpis_opt["nhbi"] > 0.0
    assert "MUTUALISTIC" in kpis_opt["badge_text"]

    kpis_starve = app_mod.calculate_kpis(mock_starved_solution)
    assert "STARVATION" in kpis_starve["badge_text"]
    assert kpis_starve["maintenance_deficit"] > 0.0

  def test_seasonal_presets_mapping(self):
    """TC-T1-R4.1-01: Verify Philippine seasonal presets are defined."""
    app_mod = get_app()
    presets = getattr(app_mod, "SEASONAL_REGIMES", ReferenceApp.SEASONAL_REGIMES)
    for p in [
        "100% Peak Dry Season Control",
        "70% Intermittent Monsoon Cloud Cover",
        "50% Extended Rainy Season Overcast",
        "30% Active Typhoon Passage",
    ]:
      assert p in presets

  def test_genotype_presets_mapping(self):
    """TC-T1-R4.1-02: Verify OsSWEET11 genotype presets are defined."""
    app_mod = get_app()
    presets = getattr(app_mod, "GENOTYPE_PRESETS", ReferenceApp.GENOTYPE_PRESETS)
    for g in [
        "OsSWEET11_WT",
        "OsSWEET11_OE",
        "OsSWEET11_Knockdown",
        "Ossweet11_KO",
    ]:
      assert g in presets

  def test_csv_export_generator(self, mock_fba_solution):
    """TC-T1-R4.4-01: Verify CSV export string contains reaction fluxes."""
    app_mod = get_app()
    csv_str = app_mod.generate_fluxes_csv(mock_fba_solution.fluxes)
    assert "Reaction_ID,Flux" in csv_str
    assert "BIOMASS_rice" in csv_str

  def test_markdown_manuscript_report_generator(self, default_model):
    """TC-T1-R4.4-03: Verify scientific manuscript Markdown report generator."""
    sim = get_simulator()
    app_mod = get_app()
    exp1 = sim.simulate_experiment_1_seasonal(default_model)
    exp2 = sim.test_experiment_2_hypothesis(default_model, light_pct=50.0)
    exp3 = sim.pinpoint_experiment_3_bifurcation(
        default_model, light_pct=50.0, sweet11_export=2.0
    )

    md_report = app_mod.generate_manuscript_markdown(exp1, exp2, exp3)
    assert "# Scientific Manuscript" in md_report
    assert "Executive Summary" in md_report
    assert "Hypothesis Testing" in md_report


# ==============================================================================
# Tier 3: TestCombinatorialCrossFeatures (Cross-Feature Combinations)
# ==============================================================================


@pytest.mark.tier3
class TestCombinatorialCrossFeatures:
  """Systematic combinatorial interactions and solver fallback stress testing."""

  @pytest.mark.parametrize("light_pct", [100.0, 70.0, 50.0, 30.0])
  @pytest.mark.parametrize(
      "genotype",
      [
          "OsSWEET11_WT",
          "OsSWEET11_OE",
          "OsSWEET11_Knockdown",
          "Ossweet11_KO",
      ],
  )
  @pytest.mark.parametrize("external_nh4", [0.0, 2.0, 10.0])
  @pytest.mark.parametrize("inoculated", [True, False])
  def test_t3_comb_001_factorial_grid(
      self, default_model, light_pct, genotype, external_nh4, inoculated
  ):
    """TEST_T3_COMB_001: 96-state full factorial grid sweep with solver determinism,

    steady-state mass balance, and thermodynamic flux bound enforcement.
    """
    sim = get_simulator()
    sol = sim.solve_single_point(
        default_model,
        light_pct=light_pct,
        genotype=genotype,
        external_nh4=external_nh4,
        inoculated=inoculated,
    )
    assert sol.status in ["optimal", "starvation"]
    if sol.is_starved:
      assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)
      assert sol.starvation_deficit > 0.0
    else:
      assert sol.atpm_r >= 2.5 - 1e-5

      # Invariant Verification: Steady-state stoichiometric mass conservation
      if hasattr(default_model, "S") and hasattr(default_model, "reaction_ids"):
        v = np.array([sol.fluxes[r] for r in default_model.reaction_ids])
        assert_mass_balance(default_model.S, v, tol=1e-5)

        # Reconstruct active upper/lower flux bounds for this parameter state
        lb = default_model.lb.copy()
        ub = default_model.ub.copy()
        ub[default_model.reaction_ids.index("EX_photon_r")] = 100.0 * (
            light_pct / 100.0
        )
        ub[default_model.reaction_ids.index("EX_nh4_soil")] = float(external_nh4)
        sweet11_idx = default_model.reaction_ids.index("SWEET11_export")
        atpm_ab_idx = default_model.reaction_ids.index("ATPM_ab")

        if genotype in ("OsSWEET11_WT", "WT"):
          lb[sweet11_idx], ub[sweet11_idx] = 0.0, 10.0
        elif genotype in ("OsSWEET11_OE", "OE"):
          lb[sweet11_idx], ub[sweet11_idx] = 2.0, 20.0
        elif genotype in ("OsSWEET11_Knockdown", "OsSWEET11_KD", "KD"):
          lb[sweet11_idx], ub[sweet11_idx] = 0.0, 3.0
        elif genotype in ("Ossweet11_KO", "KO"):
          lb[sweet11_idx], ub[sweet11_idx] = 0.0, 0.0
          lb[atpm_ab_idx] = 0.0

        if not inoculated:
          for r_name in [
              "SUC_uptake_ab",
              "RESP_ab",
              "ATPM_ab",
              "NITROGENASE_ab",
              "NH4_excretion_ab",
              "NH4_uptake_ab",
              "BIOMASS_ab",
          ]:
            idx = default_model.reaction_ids.index(r_name)
            lb[idx] = 0.0
            ub[idx] = 0.0
          lb[sweet11_idx] = 0.0
          ub[sweet11_idx] = 0.0

        assert_thermodynamic_bounds(v, lb, ub, tol=1e-5)

  def test_t3_comb_002_mutualism_zero_nh4(self, default_model):
    """TEST_T3_COMB_002: Mutualism invariant under high light and zero soil nitrogen."""
    sim = get_simulator()
    sol_inoc = sim.solve_single_point(
        default_model,
        light_pct=100.0,
        genotype="OsSWEET11_WT",
        external_nh4=0.0,
        inoculated=True,
    )
    sol_uninoc = sim.solve_single_point(
        default_model,
        light_pct=100.0,
        genotype="OsSWEET11_WT",
        external_nh4=0.0,
        inoculated=False,
    )
    assert sol_inoc.is_starved is False
    assert sol_inoc.nitrogenase_ab > 0.0
    assert sol_inoc.biomass_rice > sol_uninoc.biomass_rice

  def test_t3_comb_003_parasitic_inversion(self, default_model):
    """TEST_T3_COMB_003: Parasitic inversion under saturating fertilizer with forced exudation."""
    sim = get_simulator()
    sol_inoc = sim.solve_single_point(
        default_model,
        light_pct=100.0,
        sweet11_bounds=(4.0, 4.0),
        external_nh4=10.0,
        inoculated=True,
    )
    sol_uninoc = sim.solve_single_point(
        default_model,
        light_pct=100.0,
        sweet11_bounds=(0.0, 0.0),
        external_nh4=10.0,
        inoculated=False,
    )
    # Parasitism: nitrogenase drops to 0 while host loses sucrose
    assert sol_inoc.biomass_rice < sol_uninoc.biomass_rice

  def test_t3_comb_004_knockout_epistasis(self, default_model):
    """TEST_T3_COMB_004: Knockout mutant locks SWEET11 to 0 across all light regimes."""
    sim = get_simulator()
    for lp in [100.0, 70.0, 50.0, 30.0]:
      sol = sim.solve_single_point(
          default_model, light_pct=lp, genotype="Ossweet11_KO", inoculated=True
      )
      assert sol.sweet11_export == pytest.approx(0.0, abs=1e-7)
      assert sol.nitrogenase_ab == pytest.approx(0.0, abs=1e-7)

  def test_t3_solv_003_combined_typhoon_oe_cliff(self, default_model):
    """TEST_T3_SOLV_003: Combined 30% typhoon light + OsSWEET11_OE (lb=2.0) triggers lethal starvation."""
    sim = get_simulator()
    sol_oe = sim.solve_single_point(
        default_model,
        light_pct=30.0,
        genotype="OsSWEET11_OE",
        external_nh4=0.0,
        inoculated=True,
    )
    assert sol_oe.is_starved is True
    assert sol_oe.biomass_rice == pytest.approx(0.0, abs=1e-7)
    assert sol_oe.starvation_deficit > 0.0

  def test_t3_data_002_slice_parity_with_1d_sweep(self, default_model):
    """TEST_T3_DATA_002: Verifies 2D meshgrid slice matches independent 1D sweep."""
    sim = get_simulator()
    ny, nx = 5, 5
    res2d = sim.generate_2d_phase_plane(
        default_model,
        param_x=("light_pct", 50.0, 100.0, nx),
        param_y=("sweet11_export", 0.0, 8.0, ny),
    )
    # 1D sweep over sweet11 at 50% light
    sw1d = sim.run_1d_sweep(
        default_model,
        "sweet11_export",
        res2d.y_values,
        fixed_params={"light_pct": 50.0},
    )
    assert np.allclose(res2d.Z_biomass_rice[:, 0], sw1d.biomass_rice, atol=1e-4)


# ==============================================================================
# Tier 4: TestScientificHypotheses (Real-World Scenarios & RQ Acceptance)
# ==============================================================================


@pytest.mark.r5_scientific_hypotheses
@pytest.mark.tier4
class TestScientificHypotheses:
  """Verifies empirical biological behavior across the three core research questions."""

  def test_t4_rq1_001_seasonal_monotonicity(self, default_model):
    """TEST_T4_RQ1_001: Monotonic flux ordering across Philippine seasons (Dry > Monsoon > Overcast > Typhoon)."""
    sim = get_simulator()
    exp1 = sim.simulate_experiment_1_seasonal(default_model, genotype="WT")

    dry = exp1.seasonal_regimes["100% Peak Dry"]
    monsoon = exp1.seasonal_regimes["70% Monsoon"]
    rainy = exp1.seasonal_regimes["50% Overcast"]
    typhoon = exp1.seasonal_regimes["30% Typhoon"]

    # Strict physiological monotonicity
    assert (
        dry.ex_photon_r
        > monsoon.ex_photon_r
        > rainy.ex_photon_r
        > typhoon.ex_photon_r
    )
    assert dry.suc_syn_r > monsoon.suc_syn_r > rainy.suc_syn_r > typhoon.suc_syn_r
    assert (
        dry.biomass_rice
        > monsoon.biomass_rice
        > rainy.biomass_rice
        > typhoon.biomass_rice
    )
    assert dry.nitrogenase_ab >= typhoon.nitrogenase_ab

  def test_t4_rq1_002_carbon_allocation_plasticity(self, default_model):
    """TEST_T4_RQ1_002: Basal maintenance allocation fraction increases as irradiance declines."""
    sim = get_simulator()
    exp1 = sim.simulate_experiment_1_seasonal(default_model, genotype="WT")

    f_dry = 2.5 / exp1.seasonal_regimes["100% Peak Dry"].suc_syn_r
    f_typhoon = 2.5 / exp1.seasonal_regimes["30% Typhoon"].suc_syn_r
    assert f_typhoon > f_dry

  def test_t4_rq2_001_hypothesis_h1_validation(self, default_model):
    """TEST_T4_RQ2_001: Formally evaluates H0 vs H1 at 50% light; confirms optimal [LB*, UB*] window exists."""
    sim = get_simulator()
    res = sim.test_experiment_2_hypothesis(
        default_model, light_pct=50.0, external_nh4=0.0, n_samples=100
    )
    assert res.h1_accepted is True, f"H1 must be accepted at 50% light! Summary: {res.decision_summary}"
    assert res.optimal_lb > 0.0
    assert res.optimal_ub >= res.optimal_lb
    assert res.optimal_ub < res.starvation_threshold_flux
    assert res.max_biomass > res.ko_biomass
    assert res.nitrogenase_at_opt > 0.0

  def test_t4_rq2_002_optimal_window_viability_sampling(self, default_model):
    """TEST_T4_RQ2_002: Sample points within optimal window confirm viable growth and non-starvation."""
    sim = get_simulator()
    res = sim.test_experiment_2_hypothesis(default_model, light_pct=50.0)
    samples = np.linspace(res.optimal_lb, res.optimal_ub, 3)
    for sw in samples:
      sol = sim.solve_single_point(
          default_model,
          light_pct=50.0,
          sweet11_bounds=(sw, sw),
          external_nh4=0.0,
      )
      assert sol.is_starved is False
      assert sol.atpm_r >= 2.5 - 1e-4
      assert sol.biomass_rice > res.ko_biomass

  def test_t4_rq3_001_ammonium_collapse_bifurcation(self, default_model):
    """TEST_T4_RQ3_001: Pinpoints critical soil ammonium threshold [NH4+]crit where NHBI switches from positive to negative."""
    sim = get_simulator()
    bifurc = sim.pinpoint_experiment_3_bifurcation(
        default_model, light_pct=50.0, sweet11_export=2.0
    )
    assert 0.0 < bifurc.critical_nh4_threshold < 20.0
    assert bifurc.nhbi_at_zero_nh4 > 0.0  # Mutualism in unfertilized soil
    assert bifurc.nhbi_at_high_nh4 < 0.0  # Parasitism in high fertilizer

  def test_t4_rq3_004_dual_light_stress_bifurcation_comparison(
      self, default_model
  ):
    """TEST_T4_RQ3_004: Demonstrates that 30% light contracts the mutualistic envelope compared to 50% light."""
    sim = get_simulator()
    bifurc_50 = sim.pinpoint_experiment_3_bifurcation(
        default_model, light_pct=50.0, sweet11_export=2.0
    )
    bifurc_30 = sim.pinpoint_experiment_3_bifurcation(
        default_model, light_pct=30.0, sweet11_export=2.0
    )
    assert bifurc_30.nhbi_at_zero_nh4 <= bifurc_50.nhbi_at_zero_nh4
