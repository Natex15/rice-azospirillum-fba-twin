"""tests/test_adversarial_m1.py

Adversarial Stress Test Suite for Milestone M1 (Metabolic Model & Environment Setup).
Author: Challenger 1
Focus: Extreme boundary conditions, numerical stability, biological oracles,
robustness, and exception handling for model_builder.py.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from model_builder import (
    FBASolution,
    TwoCompartmentMetabolicModel,
    build_two_compartment_model,
    load_model_from_file,
    HAS_COBRA,
)


@pytest.fixture
def model() -> TwoCompartmentMetabolicModel:
  """Returns a clean two-compartment model instance."""
  return build_two_compartment_model()


# ==============================================================================
# Suite 1: Extreme Boundary Conditions
# ==============================================================================


class TestExtremeBoundaryConditions:
  """Adversarial challenge of extreme boundary conditions."""

  def test_night_regime_zero_light_deterministic_starvation(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Adversarial Challenge: Light = 0.0 (night regime).

    Must deterministically halt photosynthesis, trigger is_starved = True,
    set objective_value = 0.0, biomass_rice = 0.0, and record exact ATP deficit of 2.5.
    """
    model.set_light_intensity(0.0)
    model.set_genotype("WT")
    model.set_soil_ammonium(0.0)

    sol = model.optimize("BIOMASS_rice")

    assert sol.status == "starved"
    assert sol.is_starved is True
    assert sol.is_feasible is False
    assert sol.objective_value == 0.0
    assert sol.biomass_rice == 0.0
    assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-5)
    assert sol.fluxes["PHOTOSYN_r"] == 0.0
    assert sol.fluxes["SUC_SYN_r"] == 0.0
    assert sol.fluxes["RESP_r"] == 0.0
    assert sol.fluxes["ATPM_r"] == 0.0

  @pytest.mark.parametrize(
      "light,expected_deficit",
      [
          (1.0, 2.5 - (1.0 / 32.0) * 30.0),  # 2.5 - 0.9375 = 1.5625
          (2.0, 2.5 - (2.0 / 32.0) * 30.0),  # 2.5 - 1.8750 = 0.6250
      ],
  )
  def test_sub_maintenance_light_exact_slack_oracle(
      self,
      model: TwoCompartmentMetabolicModel,
      light: float,
      expected_deficit: float,
  ):
    """Adversarial Oracle: Light below maintenance threshold.

    1 sucrose requires 32 photons; 1 sucrose yields 30 ATP in root respiration.
    ATP production rate = (light / 32) * 30.
    For light < 2.6667, max ATP < 2.5, requiring exact deficit calculation.
    """
    model.set_light_intensity(light)
    model.set_genotype("WT")
    model.set_soil_ammonium(0.0)

    sol = model.optimize("BIOMASS_rice")

    assert sol.status == "starved"
    assert sol.is_starved is True
    assert sol.objective_value == 0.0
    assert sol.biomass_rice == 0.0
    assert sol.atpm_deficit == pytest.approx(expected_deficit, abs=1e-3)
    assert sol.fluxes["ATPM_r"] == pytest.approx(2.5 - expected_deficit, abs=1e-3)

  @pytest.mark.parametrize("high_light", [1000.0, 5000.0, 100000.0])
  def test_super_saturating_light_numerical_stability(
      self, model: TwoCompartmentMetabolicModel, high_light: float
  ):
    """Adversarial Challenge: Super-saturating light influx.

    Ensures no numerical overflow, solver convergence failures, or unbounded loops.
    """
    model.set_light_intensity(high_light)
    model.set_genotype("WT")
    model.set_soil_ammonium(5.0)

    sol = model.optimize("BIOMASS_rice")

    assert sol.status == "optimal"
    assert sol.is_feasible is True
    assert not np.isnan(sol.objective_value)
    assert not np.isinf(sol.objective_value)
    assert sol.objective_value > 0.0

    # Verify steady-state mass balance S * v = 0 holds
    _, S, _, _, rxn_ids, _ = model.to_scipy_lp()
    v = np.array([sol.fluxes[r_id] for r_id in rxn_ids])
    residuals = S @ v
    assert np.max(np.abs(residuals)) < 1e-7

  @pytest.mark.parametrize("light_pct", [30.0, 50.0])
  def test_ossweet11_oe_low_light_forces_host_carbon_starvation(
      self, model: TwoCompartmentMetabolicModel, light_pct: float
  ):
    """Adversarial Challenge: OsSWEET11 OE under Philippine overcast regimes (30% and 50% light).

    At 30% light, max sucrose = 30 / 32 = 0.9375 mmol/h.
    At 50% light, max sucrose = 50 / 32 = 1.5625 mmol/h.
    Enforced OE lower bound lb = 2.0 mmol/h strictly exceeds total sucrose production,
    obligating lethal carbon starvation.
    """
    model.set_light_intensity(light_pct)
    model.set_genotype("OE", lb_custom=2.0, ub_custom=20.0)
    model.set_soil_ammonium(0.0)

    sol = model.optimize("BIOMASS_rice")

    assert sol.status == "starved"
    assert sol.is_starved is True
    assert sol.objective_value == 0.0
    assert sol.biomass_rice == 0.0
    assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-3)

  def test_ossweet11_ko_complete_isolation(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Adversarial Challenge: OsSWEET11 KO knockout regime.

    Must enforce SWEET11_export == 0.0, NITROGENASE_ab == 0.0, and NH4_excretion_ab == 0.0.
    Under zero soil NH4, host rice biomass must be 0.0 (nitrogen starvation).
    Under soil NH4 = 5.0, host biomass must be positive solely from soil NH4.
    """
    model.set_light_intensity(100.0)
    model.set_genotype("KO")

    # Case A: Zero soil ammonium
    model.set_soil_ammonium(0.0)
    sol_zero = model.optimize("BIOMASS_rice")
    assert sol_zero.status == "optimal"
    assert sol_zero.sweet11_export == 0.0
    assert sol_zero.nitrogenase_ab == 0.0
    assert sol_zero.nh4_excretion_ab == 0.0
    assert sol_zero.biomass_rice == 0.0
    assert sol_zero.atpm_r >= 2.5 - 1e-5  # Maintenance satisfied, limited by N

    # Case B: Soil ammonium = 5.0
    model.set_soil_ammonium(5.0)
    sol_soil = model.optimize("BIOMASS_rice")
    assert sol_soil.status == "optimal"
    assert sol_soil.sweet11_export == 0.0
    assert sol_soil.nitrogenase_ab == 0.0
    assert sol_soil.nh4_excretion_ab == 0.0
    assert sol_soil.nh4_uptake_rice > 0.0
    assert sol_soil.biomass_rice > 0.0

  def test_soil_nh4_abundance_suppresses_nitrogenase_and_causes_parasitism(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Adversarial Challenge: High soil NH4 (> 5.0) causes diazotrophic nitrogenase shutoff

    while bacteria consume host carbon (parasitic state).

    Under zero soil NH4, the plant must invest sucrose into bacteria to fix nitrogen:
      - nitrogenase_ab > 0.0, sweet11_export > 0.0, nh4_excretion_ab > 0.0.
    Under high soil NH4 (10.0), soil NH4 uptake is energetically free, so:
      - nitrogenase_ab drops to exactly 0.0,
      - nh4_excretion_ab drops to 0.0,
      - bacteria still consume host sucrose (0.025 mmol/gDW/h) for non-growth maintenance ATPM_ab (0.5 ATP),
      - establishing the true parasitic carbon sink state!
    """
    model.set_light_intensity(100.0)
    model.set_genotype("WT")

    # Zero soil NH4 -> Active mutualism
    model.set_soil_ammonium(0.0)
    sol_zero = model.optimize("BIOMASS_rice")
    assert sol_zero.status == "optimal"
    assert sol_zero.nitrogenase_ab > 0.0
    assert sol_zero.sweet11_export > 0.0
    assert sol_zero.nh4_excretion_ab > 0.0

    # High soil NH4 -> Complete nitrogenase shutdown with host carbon consumption
    model.set_soil_ammonium(10.0)
    sol_high = model.optimize("BIOMASS_rice")
    assert sol_high.status == "optimal"
    assert sol_high.nitrogenase_ab == pytest.approx(0.0, abs=1e-6)
    assert sol_high.nh4_excretion_ab == pytest.approx(0.0, abs=1e-6)
    assert sol_high.nh4_uptake_rice > 0.0
    # Bacteria consume host sucrose to satisfy non-growth maintenance ATPM_ab = 0.5 (0.5 / 20 = 0.025 sucrose)
    assert sol_high.sweet11_export == pytest.approx(0.025, abs=1e-5)
    assert sol_high.suc_uptake_ab == pytest.approx(0.025, abs=1e-5)
    assert sol_high.atpm_ab == pytest.approx(0.5, abs=1e-5)


# ==============================================================================
# Suite 2: Robustness and Exception Handling
# ==============================================================================


class TestRobustnessAndExceptionHandling:
  """Adversarial stress-testing of input validation and error paths."""

  def test_invalid_reaction_ids_raise_keyerror(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Verify KeyError is raised when querying or setting bounds on nonexistent reactions."""
    with pytest.raises(KeyError, match="not found in model"):
      model.get_reaction_bounds("NON_EXISTENT_RXN_12345")

    with pytest.raises(KeyError, match="not found in model"):
      model.set_reaction_bounds("NON_EXISTENT_RXN_12345", 0.0, 10.0)

    with pytest.raises(KeyError, match="not found in model"):
      model.optimize(objective="NON_EXISTENT_OBJ_12345")

    with pytest.raises(KeyError):
      model.reactions.get_by_id("NON_EXISTENT_RXN_12345")

    with pytest.raises(KeyError):
      model.metabolites.get_by_id("NON_EXISTENT_MET_12345")

  def test_invalid_bounds_lb_greater_than_ub_raises_valueerror(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Verify ValueError is raised when lower bound exceeds upper bound."""
    with pytest.raises(ValueError, match="cannot exceed upper bound"):
      model.set_reaction_bounds("SWEET11_export", 10.0, 5.0)

    with pytest.raises(ValueError, match="cannot exceed upper bound"):
      model.set_reaction_bounds("ATPM_r", 100.0, 2.5)

    with pytest.raises(ValueError, match="cannot exceed upper bound"):
      model.set_reaction_bounds("EX_photon_r", 50.0, -10.0)

  @pytest.mark.parametrize(
      "invalid_genotype",
      [
          "invalid",
          "UNKNOWN_MUTANT",
          "CRISPR_SUPER_SWEET",
          "",
          "   ",
      ],
  )
  def test_invalid_genotype_string_raises_valueerror(
      self, model: TwoCompartmentMetabolicModel, invalid_genotype: str
  ):
    """Verify ValueError is raised for unmapped or malformed genotype strings."""
    with pytest.raises(ValueError, match="Unrecognized genotype profile"):
      model.set_genotype(invalid_genotype)

  def test_custom_genotype_validation(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Verify custom genotype requires both lb_custom and ub_custom, and lb <= ub."""
    with pytest.raises(ValueError, match="must be specified for 'custom'"):
      model.set_genotype("custom", lb_custom=None, ub_custom=10.0)

    with pytest.raises(ValueError, match="must be specified for 'custom'"):
      model.set_genotype("custom", lb_custom=5.0, ub_custom=None)

    with pytest.raises(ValueError, match="cannot exceed upper bound"):
      model.set_genotype("custom", lb_custom=15.0, ub_custom=5.0)

  def test_negative_environmental_inputs_handling(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Verify negative environmental inputs are clamped gracefully without crashing."""
    model.set_light_intensity(-50.0)
    lb, ub = model.get_reaction_bounds("EX_photon_r")
    assert lb == 0.0
    assert ub == 0.0

    model.set_soil_ammonium(-20.0)
    lb, ub = model.get_reaction_bounds("EX_nh4_soil")
    assert lb == 0.0
    assert ub == 0.0


# ==============================================================================
# Suite 3: File I/O Robustness (Corrupted and Empty SBML/JSON Files)
# ==============================================================================


class TestFileIORobustness:
  """Adversarial testing of SBML and JSON loaders with corrupted or malformed inputs."""

  def test_empty_string_passed_to_loader_raises_error(self):
    """Empty string passed to load_model_from_file must raise an error, not crash."""
    with pytest.raises(Exception):
      load_model_from_file("")

  def test_corrupted_json_string_raises_error(self):
    """Corrupted JSON string raises json.JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
      load_model_from_file('{"id": "Corrupted_Model", "reactions": [')

  def test_corrupted_xml_string_raises_error(self):
    """Corrupted XML string raises ET.ParseError."""
    with pytest.raises(ET.ParseError):
      load_model_from_file("<sbml><model><listOfSpecies><unclosed></sbml>")

  def test_xml_without_model_node_raises_valueerror(self):
    """Valid XML lacking a <model> element must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid SBML: no <model> element"):
      load_model_from_file("<sbml><annotation><notes>No model here</notes></annotation></sbml>")

  def test_empty_file_on_disk_raises_error(self, tmp_path: Path):
    """Empty file on disk raises an appropriate deserialization error."""
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("", encoding="utf-8")
    with pytest.raises(Exception):
      load_model_from_file(str(empty_json))

    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text("", encoding="utf-8")
    with pytest.raises(Exception):
      load_model_from_file(str(empty_xml))

  def test_corrupted_file_on_disk_raises_error(self, tmp_path: Path):
    """Corrupted file on disk raises appropriate parsing error."""
    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text('{"id": "Test", broken', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
      load_model_from_file(str(corrupt_json))

    corrupt_xml = tmp_path / "corrupt.xml"
    corrupt_xml.write_text("<?xml version='1.0'?><sbml><model>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
      load_model_from_file(str(corrupt_xml))

  def test_invalid_type_passed_to_loader_raises_typeerror(self):
    """Passing non-string, non-path, non-buffer object raises TypeError or AttributeError."""
    with pytest.raises((TypeError, AttributeError)):
      load_model_from_file(12345)


# ==============================================================================
# Suite 4: Mathematical Conservation & Thermodynamic Invariants
# ==============================================================================


class TestMathematicalAndBiologicalOracles:
  """Rigorous empirical validation of mathematical and physical conservation laws."""

  @pytest.mark.parametrize(
      "genotype,light,nh4",
      [
          ("WT", 100.0, 5.0),
          ("WT", 70.0, 2.5),
          ("OE", 100.0, 5.0),
          ("Knockdown", 100.0, 0.0),
          ("KO", 100.0, 5.0),
      ],
  )
  def test_steady_state_mass_conservation_oracle(
      self,
      model: TwoCompartmentMetabolicModel,
      genotype: str,
      light: float,
      nh4: float,
  ):
    """Empirical Oracle: Steady-state flux balance equality S * v = 0 must hold to 1e-8.

    No internal metabolite may accumulate or deplete.
    """
    model.set_light_intensity(light)
    model.set_genotype(genotype)
    model.set_soil_ammonium(nh4)

    sol = model.optimize("BIOMASS_rice")
    assert sol.status == "optimal"

    _, S, _, _, rxn_ids, _ = model.to_scipy_lp()
    v = np.array([sol.fluxes[r_id] for r_id in rxn_ids])
    residuals = S @ v

    assert np.max(np.abs(residuals)) < 1e-8

  def test_biomass_does_not_exceed_stoichiometric_theoretical_maximum(
      self, model: TwoCompartmentMetabolicModel
  ):
    """Empirical Oracle: Rice biomass yield cannot exceed stoichiometric carbon limit.

    At 100% light (100 mmol photons/h), max triose-P is 100 / 8 = 12.5.
    Max sucrose is 12.5 / 4 = 3.125 mmol sucrose.
    1 BIOMASS_rice requires 0.25 sucrose + 0.15 amino_acid (which requires 0.15 * 0.417 sucrose)
    = 0.31255 sucrose minimum just for carbon skeleton.
    Therefore, biomass cannot exceed 3.125 / 0.31255 = 9.998 gDW/h under any conditions.
    """
    model.set_light_intensity(100.0)
    model.set_genotype("WT")
    model.set_soil_ammonium(100.0)  # Excess ammonium

    sol = model.optimize("BIOMASS_rice")
    assert sol.status == "optimal"
    assert sol.biomass_rice <= 10.0 + 1e-5
