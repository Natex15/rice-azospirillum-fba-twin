"""tests/test_model_builder.py

Comprehensive Unit Test Suite for Milestone M1 (Metabolic Model & Environment Setup).
Verifies model architecture, 4-compartment topology, 24 reactions stoichiometry,
SciPy HiGHS fallback solver, numerical parity with COBRApy, two-stage carbon starvation
slack LP, genotype profiles, and pure-Python SBML / JSON round-trip serialization.
"""

import io
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.optimize import linprog

import model_builder
from model_builder import (
    FBASolution,
    TwoCompartmentMetabolicModel,
    build_two_compartment_model,
    load_model_from_file,
    HAS_COBRA,
)

if HAS_COBRA:
  import cobra


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def fresh_model():
  """Returns a freshly instantiated 4-compartment model."""
  return build_two_compartment_model()


# ==============================================================================
# 1. Topology, Compartments, and Reaction Inventory Tests
# ==============================================================================


def test_model_compartments(fresh_model):
  """Verify the 4 required biological compartments exist."""
  cmps = fresh_model.compartments
  assert "r_shoot" in cmps
  assert "r_root" in cmps
  assert "rhizo" in cmps
  assert "a_bras" in cmps
  assert len(cmps) == 4


def test_metabolite_inventory(fresh_model):
  """Verify internal metabolite pools across compartments."""
  expected_mets = [
      "photon[r_shoot]",
      "co2[r_shoot]",
      "h2o[r_shoot]",
      "o2[r_shoot]",
      "triose_p[r_shoot]",
      "sucrose[r_root]",
      "atp[r_root]",
      "nh4[r_root]",
      "amino_acid[r_root]",
      "biomass[r_root]",
      "sucrose[rhizo]",
      "nh4[rhizo]",
      "n2[rhizo]",
      "h2[rhizo]",
      "sucrose[a_bras]",
      "atp[a_bras]",
      "nh4[a_bras]",
      "biomass[a_bras]",
  ]
  for m in expected_mets:
    assert m in fresh_model.metabolites, f"Missing metabolite {m}"
  assert len(fresh_model.metabolites) == 18


def test_reaction_inventory(fresh_model):
  """Verify exact 24 reactions (16 core + 8 exchange/sink)."""
  expected_rxns = [
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
  for r in expected_rxns:
    assert r in fresh_model.reactions, f"Missing reaction {r}"
  assert len(fresh_model.reactions) == 24


def test_stoichiometric_matrix_shape(fresh_model):
  """Verify canonical LP matrix S shape is (18, 24) and matches coordinate dimensions."""
  c, S, b_eq, bounds, rxn_ids, met_ids = fresh_model.to_scipy_lp()
  assert S.shape == (18, 24)
  assert len(c) == 24
  assert len(b_eq) == 18
  assert np.all(b_eq == 0.0)
  assert len(bounds) == 24
  assert len(rxn_ids) == 24
  assert len(met_ids) == 18


# ==============================================================================
# 2. Reaction Stoichiometry & Mass Conservation
# ==============================================================================


def test_reaction_stoichiometry_details(fresh_model):
  """Verify exact coefficients for key photosynthetic, respiration, and nitrogenase reactions."""
  ps = fresh_model.reactions["PHOTOSYN_r"].stoichiometry
  assert ps["photon[r_shoot]"] == -8.0
  assert ps["co2[r_shoot]"] == -3.0
  assert ps["h2o[r_shoot]"] == -3.0
  assert ps["triose_p[r_shoot]"] == 1.0
  assert ps["o2[r_shoot]"] == 3.0

  suc_syn = fresh_model.reactions["SUC_SYN_r"].stoichiometry
  assert suc_syn["triose_p[r_shoot]"] == -4.0
  assert suc_syn["sucrose[r_root]"] == 1.0

  resp = fresh_model.reactions["RESP_r"].stoichiometry
  assert resp["sucrose[r_root]"] == -1.0
  assert resp["atp[r_root]"] == 30.0

  nif = fresh_model.reactions["NITROGENASE_ab"].stoichiometry
  assert nif["n2[rhizo]"] == -1.0
  assert nif["atp[a_bras]"] == -16.0
  assert nif["nh4[a_bras]"] == 2.0
  assert nif["h2[rhizo]"] == 1.0


# ==============================================================================
# 3. Bound Management and Genotype Profiles
# ==============================================================================


def test_default_bounds(fresh_model):
  """Verify initial physiological bounds."""
  assert fresh_model.get_reaction_bounds("ATPM_r") == (2.5, 1000.0)
  assert fresh_model.get_reaction_bounds("EX_photon_r") == (0.0, 100.0)
  assert fresh_model.get_reaction_bounds("SWEET11_export") == (0.0, 10.0)
  assert fresh_model.get_reaction_bounds("EX_nh4_soil") == (0.0, 5.0)


def test_get_set_reaction_bounds(fresh_model):
  """Verify setting and getting bounds with validation."""
  fresh_model.set_reaction_bounds("SWEET11_export", 1.5, 8.5)
  assert fresh_model.get_reaction_bounds("SWEET11_export") == (1.5, 8.5)

  with pytest.raises(ValueError):
    fresh_model.set_reaction_bounds("SWEET11_export", 10.0, 5.0)

  with pytest.raises(KeyError):
    fresh_model.set_reaction_bounds("NON_EXISTENT_RXN", 0.0, 1.0)


@pytest.mark.parametrize(
    "genotype,exp_lb,exp_ub",
    [
        ("WT", 0.0, 10.0),
        ("OsSWEET11_WT", 0.0, 10.0),
        ("OE", 2.0, 20.0),
        ("OsSWEET11_OE", 2.0, 20.0),
        ("Knockdown", 0.0, 3.0),
        ("OsSWEET11_Knockdown", 0.0, 3.0),
        ("KO", 0.0, 0.0),
        ("Ossweet11_KO", 0.0, 0.0),
    ],
)
def test_genotype_parameterization(fresh_model, genotype, exp_lb, exp_ub):
  """Verify preset genotype profile bounds on SWEET11_export."""
  fresh_model.set_genotype(genotype)
  lb, ub = fresh_model.get_reaction_bounds("SWEET11_export")
  assert lb == pytest.approx(exp_lb, abs=1e-3)
  assert ub == pytest.approx(exp_ub, abs=1e-3)


def test_custom_genotype(fresh_model):
  """Verify custom genotype setting."""
  fresh_model.set_genotype("custom", lb_custom=1.2, ub_custom=6.7)
  assert fresh_model.get_reaction_bounds("SWEET11_export") == (1.2, 6.7)

  with pytest.raises(ValueError):
    fresh_model.set_genotype("custom", lb_custom=None, ub_custom=5.0)


def test_environmental_setters(fresh_model):
  """Verify set_light_intensity and set_soil_ammonium."""
  fresh_model.set_light_intensity(70.0)
  assert fresh_model.get_reaction_bounds("EX_photon_r") == (0.0, 70.0)

  fresh_model.set_soil_ammonium(3.2)
  assert fresh_model.get_reaction_bounds("EX_nh4_soil") == (0.0, 3.2)


# ==============================================================================
# 4. SciPy HiGHS Solver & Two-Stage Starvation Detection
# ==============================================================================


def test_optimal_fba_solution(fresh_model):
  """Verify baseline FBA optimization under 100% light."""
  fresh_model.set_light_intensity(100.0)
  fresh_model.set_genotype("WT")
  fresh_model.set_soil_ammonium(0.0)  # Pure mutualism

  sol = fresh_model.optimize()
  assert sol.status == "optimal"
  assert sol.is_feasible is True
  assert sol.is_starved is False
  assert sol.objective_value > 4.0
  assert sol.biomass_rice > 4.0
  assert sol.atpm_deficit == 0.0
  assert sol.atpm_r >= 2.5 - 1e-5


def test_two_stage_carbon_starvation_zero_light(fresh_model):
  """Verify zero light triggers lethal carbon starvation with full ATPM deficit."""
  fresh_model.set_light_intensity(0.0)

  sol = fresh_model.optimize()
  assert sol.status == "starved"
  assert sol.is_feasible is False
  assert sol.is_starved is True
  assert sol.objective_value == 0.0
  assert sol.biomass_rice == 0.0
  assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-4)


def test_forced_oe_starvation_low_light(fresh_model):
  """Verify forced SWEET11 overexpression under low light (50%) causes starvation (H1)."""
  fresh_model.set_light_intensity(50.0)
  # Photosynthetic sucrose at 50% light is 50/(8*4) = 1.5625 mmol/h
  # Forcing export of 2.0 mmol/h depletes all root carbon and starves plant
  fresh_model.set_genotype("OE", lb_custom=2.0, ub_custom=20.0)
  fresh_model.set_soil_ammonium(0.0)

  sol = fresh_model.optimize()
  assert sol.is_starved is True
  assert sol.status == "starved"
  assert sol.objective_value == 0.0
  assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-3)


# ==============================================================================
# 5. Numerical Parity between SciPy HiGHS and COBRApy
# ==============================================================================


@pytest.mark.skipif(not HAS_COBRA, reason="COBRApy not installed")
def test_cobra_parity(fresh_model):
  """Verify 100% numerical parity between pure SciPy HiGHS and COBRApy solver."""
  fresh_model.set_light_intensity(100.0)
  fresh_model.set_soil_ammonium(5.0)

  scipy_sol = fresh_model.optimize()

  cobra_model = fresh_model.to_cobra()
  cobra_sol = cobra_model.optimize()

  assert scipy_sol.status == "optimal"
  assert cobra_sol.status == "optimal"
  assert abs(scipy_sol.objective_value - cobra_sol.objective_value) < 1e-6

  for rxn_id, scipy_flux in scipy_sol.fluxes.items():
    cobra_flux = cobra_sol.fluxes[rxn_id]
    assert abs(scipy_flux - cobra_flux) < 1e-4, f"Flux mismatch in {rxn_id}"


# ==============================================================================
# 6. SBML and JSON Serialization Round-Trip Tests
# ==============================================================================


def test_json_roundtrip(fresh_model, tmp_path):
  """Verify export and reload via pure-Python JSON preserves topology and solution."""
  fresh_model.set_reaction_bounds("SWEET11_export", 1.25, 7.5)
  json_file = str(tmp_path / "model_roundtrip.json")

  fresh_model.to_json(json_file)
  assert os.path.exists(json_file)

  reloaded = TwoCompartmentMetabolicModel.from_json(json_file)
  assert len(reloaded.compartments) == len(fresh_model.compartments)
  assert len(reloaded.metabolites) == len(fresh_model.metabolites)
  assert len(reloaded.reactions) == len(fresh_model.reactions)
  assert reloaded.get_reaction_bounds("SWEET11_export") == (1.25, 7.5)

  # Check solution identity
  sol_orig = fresh_model.optimize()
  sol_reload = reloaded.optimize()
  assert sol_orig.status == sol_reload.status
  assert abs(sol_orig.objective_value - sol_reload.objective_value) < 1e-6


def test_sbml_roundtrip(fresh_model, tmp_path):
  """Verify export and reload via pure-Python SBML L3V1 FBCv2 XML."""
  fresh_model.set_reaction_bounds("SWEET11_export", 0.5, 9.0)
  sbml_file = str(tmp_path / "model_roundtrip.xml")

  fresh_model.to_sbml(sbml_file)
  assert os.path.exists(sbml_file)

  # Verify valid XML structure
  tree = ET.parse(sbml_file)
  root = tree.getroot()
  assert "sbml" in root.tag.lower()

  reloaded = TwoCompartmentMetabolicModel.from_sbml(sbml_file)
  assert len(reloaded.compartments) == 4
  assert len(reloaded.reactions) == 24
  assert len(reloaded.metabolites) == 18
  assert reloaded.get_reaction_bounds("SWEET11_export") == (0.5, 9.0)

  sol_orig = fresh_model.optimize()
  sol_reload = reloaded.optimize()
  assert sol_orig.status == sol_reload.status
  assert abs(sol_orig.objective_value - sol_reload.objective_value) < 1e-6


def test_load_model_from_file_universal(fresh_model, tmp_path):
  """Verify universal loader handles both JSON and SBML paths and buffers."""
  json_path = str(tmp_path / "test.json")
  sbml_path = str(tmp_path / "test.xml")

  fresh_model.to_json(json_path)
  fresh_model.to_sbml(sbml_path)

  # From file paths
  m_json = load_model_from_file(json_path)
  assert len(m_json.reactions) == 24

  m_sbml = load_model_from_file(sbml_path)
  assert len(m_sbml.reactions) == 24

  # From string buffers
  with open(json_path, "r", encoding="utf-8") as f:
    buf_json = io.StringIO(f.read())
  m_buf_json = load_model_from_file(buf_json)
  assert len(m_buf_json.reactions) == 24

  with open(sbml_path, "r", encoding="utf-8") as f:
    buf_sbml = io.StringIO(f.read())
  m_buf_sbml = load_model_from_file(buf_sbml)
  assert len(m_buf_sbml.reactions) == 24
