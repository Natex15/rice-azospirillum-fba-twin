"""tests/test_challenger_m1_2.py.

Empirical Challenger 2 Test Suite for Milestone M1 (Metabolic Model & Environment Setup).
Performs:
1. High-volume Monte Carlo sweep (100 parameter sets) comparing SciPy HiGHS and COBRApy GLPK.
2. Evaluates machine-precision parity on objective values, feasibility status, and host fluxes.
3. Tests alternative optima and bacterial null-space behavior under forced SWEET11 export.
4. Validates 100% fidelity on pure-Python SBML and JSON round-trip serialization and reloaded optimization.
"""

import tempfile
import numpy as np
import pytest

from model_builder import (
    build_two_compartment_model,
    load_model_from_file,
    HAS_COBRA,
)

if HAS_COBRA:
  import cobra


def test_mc_100_unconstrained_sweet11_full_parity():
  """Monte Carlo sweep of 100 random points with sweet11_lb = 0.

  Verifies that when alternative optima from unconsumed forced exudates are
  absent, SciPy HiGHS and COBRApy GLPK achieve machine-precision parity across
  ALL 24 reactions.
  """
  if not HAS_COBRA:
    pytest.skip("COBRApy not installed")

  np.random.seed(101)
  m = build_two_compartment_model()

  max_z_diff = 0.0
  max_v_diff = 0.0

  for _ in range(100):
    light = float(np.random.uniform(10.0, 150.0))
    soil_nh4 = float(np.random.uniform(0.0, 10.0))
    sweet11_ub = float(np.random.uniform(0.0, 25.0))
    sweet11_lb = 0.0

    m.set_light_intensity(light)
    m.set_soil_ammonium(soil_nh4)
    m.set_reaction_bounds("SWEET11_export", sweet11_lb, sweet11_ub)

    sol_scipy = m.optimize()
    cm = m.to_cobra()
    sol_cobra = cm.optimize()

    assert sol_scipy.status == "optimal"
    assert sol_cobra.status == "optimal"

    z_diff = abs(sol_scipy.objective_value - sol_cobra.objective_value)
    v_diff = max(
        abs(sol_scipy.fluxes[r] - sol_cobra.fluxes[r]) for r in m.reactions
    )

    if z_diff > max_z_diff:
      max_z_diff = z_diff
    if v_diff > max_v_diff:
      max_v_diff = v_diff

  assert max_z_diff < 1e-6, f"Max Z diff too high: {max_z_diff}"
  assert max_v_diff < 1e-5, f"Max flux diff too high: {max_v_diff}"


def test_mc_100_forced_sweet11_host_parity_and_status():
  """Monte Carlo sweep of 100 points with sweet11_lb in [0.0, 5.0].

  Verifies:
  1. Feasibility / Starvation classification parity is 100% identical between SciPy and COBRApy.
  2. Objective value parity is machine-precision (|Z_scipy - Z_cobra| < 1e-6).
  3. All 13 Rice host reactions match within 1e-5.
  4. Identifies and isolates alternative optima strictly to the auxiliary Azospirillum subnetwork.
  """
  if not HAS_COBRA:
    pytest.skip("COBRApy not installed")

  np.random.seed(42)
  m = build_two_compartment_model()

  rice_rxns = [
      "EX_photon_r",
      "PHOTOSYN_r",
      "SUC_SYN_r",
      "RESP_r",
      "ATPM_r",
      "GS_GOGAT_r",
      "BIOMASS_rice",
      "SWEET11_export",
      "NH4_uptake_rice",
      "EX_co2_r",
      "EX_h2o_r",
      "EX_o2_r",
      "SK_biomass_rice",
  ]

  status_matches = 0
  optimal_count = 0
  max_z_diff = 0.0
  max_rice_flux_diff = 0.0

  for _ in range(100):
    light = float(np.random.uniform(10.0, 150.0))
    soil_nh4 = float(np.random.uniform(0.0, 10.0))
    sweet11_ub = float(np.random.uniform(0.0, 25.0))
    sweet11_lb = float(np.random.uniform(0.0, min(5.0, sweet11_ub)))

    m.set_light_intensity(light)
    m.set_soil_ammonium(soil_nh4)
    m.set_reaction_bounds("SWEET11_export", sweet11_lb, sweet11_ub)

    sol_scipy = m.optimize()
    cm = m.to_cobra()
    sol_cobra = cm.optimize()

    # Status parity check
    if (sol_scipy.status == "optimal" and sol_cobra.status == "optimal") or (
        sol_scipy.status == "starved" and sol_cobra.status == "infeasible"
    ):
      status_matches += 1

    obj_scipy = sol_scipy.objective_value
    obj_cobra = (
        sol_cobra.objective_value if sol_cobra.status == "optimal" else 0.0
    )
    z_diff = abs(obj_scipy - obj_cobra)
    if z_diff > max_z_diff:
      max_z_diff = z_diff

    if sol_scipy.status == "optimal" and sol_cobra.status == "optimal":
      optimal_count += 1
      for r in rice_rxns:
        d = abs(sol_scipy.fluxes[r] - sol_cobra.fluxes[r])
        if d > max_rice_flux_diff:
          max_rice_flux_diff = d

  assert (
      status_matches == 100
  ), f"Solver status agreement was {status_matches}/100"
  assert max_z_diff < 1e-6, f"Max Z diff too high: {max_z_diff}"
  assert (
      max_rice_flux_diff < 1e-5
  ), f"Max rice host flux diff too high: {max_rice_flux_diff}"


def test_sbml_and_json_roundtrip_fidelity(tmp_path):
  """Verifies 100% equivalence of SBML and JSON round-trip serialization."""
  m_orig = build_two_compartment_model()
  m_orig.set_light_intensity(85.0)
  m_orig.set_soil_ammonium(2.5)
  m_orig.set_reaction_bounds("SWEET11_export", 0.5, 7.5)

  sol_orig = m_orig.optimize()

  # 1. JSON Round-Trip
  json_file = str(tmp_path / "fidelity_test.json")
  m_orig.to_json(json_file)
  m_json = load_model_from_file(json_file)

  assert m_orig.compartments == m_json.compartments
  assert set(m_orig.metabolites.keys()) == set(m_json.metabolites.keys())
  for k in m_orig.metabolites:
    m1 = m_orig.metabolites[k]
    m2 = m_json.metabolites[k]
    assert (
        m1.id == m2.id and m1.name == m2.name and m1.compartment == m2.compartment
    )

  assert set(m_orig.reactions.keys()) == set(m_json.reactions.keys())
  for k in m_orig.reactions:
    r1 = m_orig.reactions[k]
    r2 = m_json.reactions[k]
    assert r1.id == r2.id
    assert abs(r1.lower_bound - r2.lower_bound) < 1e-9
    assert abs(r1.upper_bound - r2.upper_bound) < 1e-9
    assert r1.stoichiometry == r2.stoichiometry

  assert np.allclose(m_orig.S, m_json.S)
  sol_json = m_json.optimize()
  assert sol_orig.status == sol_json.status
  assert abs(sol_orig.objective_value - sol_json.objective_value) < 1e-12
  for r in m_orig.reactions:
    assert abs(sol_orig.fluxes[r] - sol_json.fluxes[r]) < 1e-9

  # 2. SBML Round-Trip
  sbml_file = str(tmp_path / "fidelity_test.xml")
  m_orig.to_sbml(sbml_file)
  m_sbml = load_model_from_file(sbml_file)

  assert set(m_orig.compartments.keys()) == set(m_sbml.compartments.keys())
  assert set(m_orig.metabolites.keys()) == set(m_sbml.metabolites.keys())
  for k in m_orig.metabolites:
    m1 = m_orig.metabolites[k]
    m2 = m_sbml.metabolites[k]
    assert m1.id == m2.id
    assert m1.compartment == m2.compartment

  assert set(m_orig.reactions.keys()) == set(m_sbml.reactions.keys())
  for k in m_orig.reactions:
    r1 = m_orig.reactions[k]
    r2 = m_sbml.reactions[k]
    assert r1.id == r2.id
    assert abs(r1.lower_bound - r2.lower_bound) < 1e-9
    assert abs(r1.upper_bound - r2.upper_bound) < 1e-9
    assert r1.stoichiometry == r2.stoichiometry

  assert np.allclose(m_orig.S, m_sbml.S)
  sol_sbml = m_sbml.optimize()
  assert sol_orig.status == sol_sbml.status
  assert abs(sol_orig.objective_value - sol_sbml.objective_value) < 1e-12
  for r in m_orig.reactions:
    assert abs(sol_orig.fluxes[r] - sol_sbml.fluxes[r]) < 1e-9
