"""tests/test_adversarial_m4_1.py.

Adversarial Stress Testing Suite for Streamlit Web Application (app.py)
Milestone M4: Challenger 1 Empirical Verification Suite.

Validates:
1. Parameter boundary stress testing (Light, SWEET11 bounds, Soil NH4+, Inverted bounds, Extremes)
2. Model uploader resilience (Empty SBML, Corrupt XML/JSON, Missing reactions, Binary garbage)
3. KPI & Status badge stress testing (Zero sucrose, Extreme starvation, Parasitism transitions, Mock objects)
4. Data exporters and visualization resilience under edge inputs.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import app
import model_builder
from model_builder import (
    FBASolution as ModelFBASolution,
    TwoCompartmentMetabolicModel,
    build_two_compartment_model,
    load_model_from_file,
)
import simulation_engine
from simulation_engine import (
    FBASolution as SimFBASolution,
    solve_single_point,
    solve_uninoculated_baseline,
)
import visualizer_3d


# ==============================================================================
# 1. Parameter Boundary Stress Tests
# ==============================================================================


class TestParameterBoundaries:
    """Stress tests parameter boundaries, inverted ranges, and numerical extremes."""

    @pytest.fixture(scope="class")
    def default_model(self) -> TwoCompartmentMetabolicModel:
        return app.get_default_model()

    @pytest.mark.parametrize("light_val", [0.0, 150.0, -10.0, -100.0, -1e-6])
    def test_light_boundaries_single_point(
        self, default_model: TwoCompartmentMetabolicModel, light_val: float
    ):
        """Tests that solve_single_point handles 0%, 150%, and negative light gracefully."""
        sol = solve_single_point(
            model=default_model,
            light_pct=light_val,
            genotype="OsSWEET11_WT",
            external_nh4=0.0,
            inoculated=True,
        )
        assert isinstance(sol, (SimFBASolution, ModelFBASolution))
        assert sol.status in ("optimal", "starvation", "infeasible")
        assert not math.isnan(sol.biomass_rice)
        assert not math.isnan(sol.nitrogenase_ab)
        assert not math.isnan(sol.sweet11_export)

        if light_val <= 0.0:
            # Under zero or negative irradiance, photosynthetic flux is 0,
            # which cannot satisfy ATPM_r >= 2.5, triggering starvation
            assert sol.is_starved is True
            assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)
            assert sol.starvation_deficit > 0.0
        elif light_val >= 100.0:
            # High light should allow positive vegetative growth
            assert sol.is_starved is False
            assert sol.biomass_rice > 0.05

    @pytest.mark.parametrize("light_val", [0.0, 150.0, -20.0, 1e4, 1e-8])
    def test_light_boundaries_fba_sensitivities(
        self, default_model: TwoCompartmentMetabolicModel, light_val: float
    ):
        """Tests simulate_fba_with_sensitivities under boundary and extreme light inputs."""
        res = app.simulate_fba_with_sensitivities(
            model=default_model,
            light_pct=light_val,
            genotype="OsSWEET11_WT",
            external_nh4=0.0,
            inoculated=True,
        )
        assert isinstance(res, dict)
        assert "status" in res
        assert "biomass_rice" in res
        assert "fluxes" in res
        assert "shadow_prices" in res
        assert "reduced_costs" in res
        assert not math.isnan(res["biomass_rice"])

        if light_val <= 0.0:
            assert res["is_starved"] is True
            assert res["biomass_rice"] == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize(
        "lb,ub,desc",
        [
            (0.0, 0.0, "Complete Knockout Ossweet11_KO"),
            (20.0, 20.0, "Extreme Constitutive Overexpression"),
            (0.0, 25.0, "Maximal Upper Bound"),
            (15.0, 5.0, "Inverted Bounds LB > UB (Adversarial)"),
            (10.0, 0.0, "Strict Inversion LB > UB (Adversarial)"),
            (-5.0, 10.0, "Negative Lower Bound (Adversarial)"),
        ],
    )
    def test_sweet11_boundary_combinations(
        self,
        default_model: TwoCompartmentMetabolicModel,
        lb: float,
        ub: float,
        desc: str,
    ):
        """Stress tests SWEET11 bounds including inverted (LB > UB) and extreme boundaries."""
        # Test in solve_single_point
        sol = solve_single_point(
            model=default_model,
            light_pct=50.0,
            sweet11_bounds=(lb, ub),
            external_nh4=0.0,
            inoculated=True,
        )
        assert isinstance(sol, (SimFBASolution, ModelFBASolution)), f"Failed on {desc}"
        assert not math.isnan(sol.biomass_rice)
        assert not math.isnan(sol.nitrogenase_ab)

        # Inverted bounds (lb > ub) must be handled gracefully without unhandled exception
        if lb > ub:
            assert sol.is_starved is True
            assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)

        # Test in simulate_fba_with_sensitivities
        sens_res = app.simulate_fba_with_sensitivities(
            model=default_model,
            light_pct=50.0,
            sweet11_bounds=(lb, ub),
            external_nh4=0.0,
            inoculated=True,
        )
        assert isinstance(sens_res, dict)
        if lb > ub:
            assert sens_res["status"] == "infeasible"
            assert sens_res["biomass_rice"] == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("nh4_val", [0.0, 50.0, 100.0, -10.0, 1e4, 1e-6])
    def test_soil_nh4_boundaries(
        self, default_model: TwoCompartmentMetabolicModel, nh4_val: float
    ):
        """Stress tests external soil ammonium across zero, high, negative, and extremes."""
        sol = solve_single_point(
            model=default_model,
            light_pct=100.0,
            genotype="OsSWEET11_WT",
            external_nh4=nh4_val,
            inoculated=True,
        )
        assert isinstance(sol, (SimFBASolution, ModelFBASolution))
        assert not math.isnan(sol.biomass_rice)
        assert not math.isnan(sol.nitrogenase_ab)

        if nh4_val >= 50.0:
            # Saturated external NH4+ represses diazotrophic nitrogenase
            assert sol.nitrogenase_ab == pytest.approx(0.0, abs=1e-4)

    def test_visualizations_under_boundary_states(
        self, default_model: TwoCompartmentMetabolicModel
    ):
        """Verifies 3D WebGL and Plotly visualizers generate valid outputs under edge states."""
        # 1. Total Darkness Starvation State
        sol_dark = solve_single_point(
            model=default_model,
            light_pct=0.0,
            genotype="OsSWEET11_WT",
            external_nh4=0.0,
        )
        html_dark = visualizer_3d.render_threejs_rhizosphere(
            light_pct=sol_dark.ex_photon_r,
            sweet11_flux=sol_dark.sweet11_export,
            nh4_flux=sol_dark.nitrogenase_ab,
            is_starved=sol_dark.is_starved,
            is_parasitic=True,
            height=600,
        )
        assert isinstance(html_dark, str)
        assert "THREE.WebGLRenderer" in html_dark
        assert len(html_dark) > 500

        fig_dark = visualizer_3d.plot_3d_anatomical_scene(
            fluxes=sol_dark.fluxes,
            is_starved=sol_dark.is_starved,
            is_parasitic=True,
            show_soil_cutaway=True,
        )
        assert fig_dark is not None
        assert len(fig_dark.data) > 0

        # 2. Extreme High Light Saturated State
        sol_high = solve_single_point(
            model=default_model,
            light_pct=150.0,
            genotype="OsSWEET11_WT",
            external_nh4=100.0,
        )
        html_high = visualizer_3d.render_threejs_rhizosphere(
            light_pct=sol_high.ex_photon_r,
            sweet11_flux=sol_high.sweet11_export,
            nh4_flux=sol_high.nitrogenase_ab,
            is_starved=sol_high.is_starved,
            is_parasitic=True,
            height=600,
        )
        assert isinstance(html_high, str)
        assert "THREE.WebGLRenderer" in html_high

        fig_high = visualizer_3d.plot_3d_anatomical_scene(
            fluxes=sol_high.fluxes,
            is_starved=sol_high.is_starved,
            is_parasitic=True,
            show_soil_cutaway=False,
        )
        assert fig_high is not None


# ==============================================================================
# 2. Model Uploader Resilience Tests
# ==============================================================================


class TestModelUploaderResilience:
    """Stress tests model loading and uploader validation logic against adversarial inputs."""

    def test_uploader_empty_file_bytes(self):
        """Verifies empty file buffer triggers clean exception and fallback."""
        empty_buf = io.BytesIO(b"")
        with pytest.raises(Exception):
            load_model_from_file(empty_buf)

    def test_uploader_corrupt_xml(self):
        """Verifies malformed XML triggers XML parse error cleanly."""
        corrupt_xml = io.BytesIO(
            b'<?xml version="1.0" encoding="UTF-8"?><sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"><model id="incomplete">'
        )
        with pytest.raises(Exception):
            load_model_from_file(corrupt_xml)

    def test_uploader_corrupt_json(self):
        """Verifies malformed JSON triggers JSONDecodeError cleanly."""
        corrupt_json = io.BytesIO(b'{"reactions": [{"id": "EX_photon_r", "bounds": [0, 100], truncated')
        with pytest.raises(Exception):
            load_model_from_file(corrupt_json)

    def test_uploader_non_model_json(self):
        """Verifies JSON that is valid JSON but lacks metabolic structure triggers warning and fallback."""
        non_model_json = io.BytesIO(
            json.dumps({"greeting": "hello world", "data": [1, 2, 3]}).encode("utf-8")
        )
        try:
            m = load_model_from_file(non_model_json)
            # If from_json parsed it into a blank model, app.py check must flag all missing reactions
            rxn_names = (
                m.reaction_ids
                if hasattr(m, "reaction_ids")
                else list(m.reactions.keys())
            )
            required_rxns = ["EX_photon_r", "SWEET11_export", "NITROGENASE_ab"]
            missing = [r for r in required_rxns if r not in rxn_names]
            assert len(missing) == 3, f"Expected all required reactions missing, got {missing}"
        except Exception:
            pass  # Raising an exception is also acceptable and caught by app.py

    def test_uploader_binary_garbage(self):
        """Verifies arbitrary random binary bytes trigger clean exception."""
        garbage = io.BytesIO(bytes([i % 256 for i in range(512)]))
        with pytest.raises(Exception):
            load_model_from_file(garbage)

    def test_uploader_missing_essential_reactions_detection(self):
        """Verifies model with missing essential reactions triggers warning branch in app.py logic."""
        # Create a valid JSON model missing NITROGENASE_ab and SWEET11_export
        partial_model_dict = {
            "id": "PartialModel",
            "reactions": [
                {
                    "id": "EX_photon_r",
                    "name": "Photon Influx",
                    "stoichiometry": {"photon[r_shoot]": 1.0},
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                },
                {
                    "id": "BIOMASS_rice",
                    "name": "Rice Biomass",
                    "stoichiometry": {"photon[r_shoot]": -1.0},
                    "lower_bound": 0.0,
                    "upper_bound": 1000.0,
                    "objective_coefficient": 1.0,
                },
            ],
            "metabolites": [
                {"id": "photon[r_shoot]", "compartment": "r_shoot"},
            ],
        }
        partial_buf = io.BytesIO(json.dumps(partial_model_dict).encode("utf-8"))
        loaded_model = load_model_from_file(partial_buf)
        assert loaded_model is not None

        # Replicate app.py validation logic:
        required_rxns = ["EX_photon_r", "SWEET11_export", "NITROGENASE_ab"]
        rxn_names = (
            loaded_model.reaction_ids
            if hasattr(loaded_model, "reaction_ids")
            else list(loaded_model.reactions.keys())
        )
        missing = [r for r in required_rxns if r not in rxn_names]

        # Ensure missing reactions are identified
        assert "SWEET11_export" in missing
        assert "NITROGENASE_ab" in missing
        assert "EX_photon_r" not in missing
        assert len(missing) == 2

    def test_uploader_fallback_simulation_integrity(self):
        """Verifies default model remains intact and operational following upload failure."""
        default_model = app.get_default_model()
        sol = solve_single_point(default_model, light_pct=70.0, external_nh4=5.0)
        assert sol.is_feasible is True
        assert sol.biomass_rice > 0.0


# ==============================================================================
# 3. KPI & Status Badge Stress Testing
# ==============================================================================


class TestKPICalculationRobustness:
    """Stress tests calculate_kpis across biological edge cases, zero denominators, and mock objects."""

    def test_kpis_zero_sucrose_synthesis(self):
        """Verifies zero sucrose synthesis does not raise ZeroDivisionError and yields 0.0% export."""
        class MockSolZeroSuc:
            biomass_rice = 0.10
            suc_syn_r = 0.0
            sweet11_export = 0.0
            is_starved = False
            nitrogenase_ab = 0.02
            starvation_deficit = 0.0

        class MockUninoc:
            biomass_rice = 0.08

        kpis = app.calculate_kpis(MockSolZeroSuc(), MockUninoc())
        assert kpis["sucrose_export_pct"] == 0.0
        assert kpis["nhbi"] == pytest.approx(0.02, abs=1e-5)
        assert kpis["badge_text"] == "MUTUALISTIC COOPERATION"
        assert kpis["badge_class"] == "success"

    def test_kpis_zero_synthesis_with_forced_export(self):
        """Verifies forced SWEET11 export with zero synthesis avoids ZeroDivisionError."""
        class MockForcedExport:
            biomass_rice = 0.0
            suc_syn_r = 0.0
            sweet11_export = 2.0
            is_starved = True
            nitrogenase_ab = 0.0
            starvation_deficit = 1.8

        kpis = app.calculate_kpis(MockForcedExport())
        assert kpis["sucrose_export_pct"] == 0.0
        assert kpis["badge_text"] == "STARVATION COLLAPSE"
        assert kpis["badge_class"] == "error"

    def test_kpis_extreme_starvation_deficit(self):
        """Verifies extreme maintenance deficit sets correct badge and maintenance_deficit."""
        class MockSevereStarve:
            biomass_rice = 0.0
            suc_syn_r = 0.0
            sweet11_export = 10.0
            is_starved = True
            nitrogenase_ab = 0.0
            starvation_deficit = 999.5

        kpis = app.calculate_kpis(MockSevereStarve())
        assert kpis["maintenance_deficit"] == 999.5
        assert kpis["badge_text"] == "STARVATION COLLAPSE"
        assert kpis["badge_class"] == "error"

    def test_kpis_mutualism_transition(self):
        """Verifies positive NHBI and positive nitrogenase sets MUTUALISTIC COOPERATION."""
        class MockInoc:
            biomass_rice = 0.12
            suc_syn_r = 5.0
            sweet11_export = 1.0
            is_starved = False
            nitrogenase_ab = 0.08
            starvation_deficit = 0.0

        class MockUninoc:
            biomass_rice = 0.08

        kpis = app.calculate_kpis(MockInoc(), MockUninoc())
        assert kpis["nhbi"] == pytest.approx(0.04, abs=1e-5)
        assert kpis["sucrose_export_pct"] == pytest.approx(20.0, abs=1e-3)
        assert kpis["badge_text"] == "MUTUALISTIC COOPERATION"
        assert kpis["badge_class"] == "success"

    def test_kpis_parasitism_negative_nhbi(self):
        """Verifies negative NHBI sets PARASITIC CARBON DRAIN badge even if nitrogenase > 0."""
        class MockInocDepressed:
            biomass_rice = 0.06  # less than uninoculated 0.08
            suc_syn_r = 4.0
            sweet11_export = 1.5
            is_starved = False
            nitrogenase_ab = 0.02
            starvation_deficit = 0.0

        class MockUninoc:
            biomass_rice = 0.08

        kpis = app.calculate_kpis(MockInocDepressed(), MockUninoc())
        assert kpis["nhbi"] == pytest.approx(-0.02, abs=1e-5)
        assert kpis["badge_text"] == "PARASITIC CARBON DRAIN"
        assert kpis["badge_class"] == "error"

    def test_kpis_parasitism_zero_nitrogenase(self):
        """Verifies zero nitrogenase sets PARASITIC CARBON DRAIN badge even if nhbi >= 0."""
        class MockInocZeroNif:
            biomass_rice = 0.09  # slightly greater than 0.08
            suc_syn_r = 4.0
            sweet11_export = 1.0
            is_starved = False
            nitrogenase_ab = 0.0  # nitrogenase shutoff by NH4+
            starvation_deficit = 0.0

        class MockUninoc:
            biomass_rice = 0.08

        kpis = app.calculate_kpis(MockInocZeroNif(), MockUninoc())
        assert kpis["nhbi"] == pytest.approx(0.01, abs=1e-5)
        assert kpis["badge_text"] == "PARASITIC CARBON DRAIN"
        assert kpis["badge_class"] == "error"

    def test_kpis_nitrogenase_threshold_boundary(self):
        """Tests exact 1e-4 boundary for nitrogenase activity in calculate_kpis."""
        class MockInocBorderlinePass:
            biomass_rice = 0.09
            suc_syn_r = 4.0
            sweet11_export = 1.0
            is_starved = False
            nitrogenase_ab = 1.0001e-4
            starvation_deficit = 0.0

        class MockInocBorderlineFail:
            biomass_rice = 0.09
            suc_syn_r = 4.0
            sweet11_export = 1.0
            is_starved = False
            nitrogenase_ab = 0.9999e-4
            starvation_deficit = 0.0

        class MockUninoc:
            biomass_rice = 0.08

        kpis_pass = app.calculate_kpis(MockInocBorderlinePass(), MockUninoc())
        assert kpis_pass["badge_text"] == "MUTUALISTIC COOPERATION"

        kpis_fail = app.calculate_kpis(MockInocBorderlineFail(), MockUninoc())
        assert kpis_fail["badge_text"] == "PARASITIC CARBON DRAIN"

    def test_kpis_none_solution_inputs(self):
        """Verifies calculate_kpis handles None inputs gracefully without crashing."""
        kpis = app.calculate_kpis(None, None)
        assert isinstance(kpis, dict)
        assert "nhbi" in kpis
        assert "sucrose_export_pct" in kpis
        assert "maintenance_deficit" in kpis
        assert "badge_text" in kpis
        assert "badge_class" in kpis

    def test_kpis_duck_typing_missing_attributes(self):
        """Verifies calculate_kpis works on generic dict-like or empty objects."""
        class EmptyObj:
            pass

        kpis = app.calculate_kpis(EmptyObj(), EmptyObj())
        assert isinstance(kpis, dict)
        assert kpis["nhbi"] == pytest.approx(-0.08, abs=1e-5)
        assert kpis["sucrose_export_pct"] == 0.0
        assert kpis["maintenance_deficit"] == 0.0


# ==============================================================================
# 4. Data Exporters & Manuscript Generator Stress Tests
# ==============================================================================


class TestDataExporters:
    """Stress tests CSV and Markdown generation functions."""

    def test_generate_fluxes_csv_empty(self):
        """Verifies empty flux dictionary outputs valid CSV headers."""
        csv_str = app.generate_fluxes_csv({})
        assert "Reaction_ID,Flux" in csv_str

    def test_generate_fluxes_csv_populated(self):
        """Verifies standard flux dictionary produces parseable CSV."""
        fluxes = {
            "EX_photon_r": 100.0,
            "SUC_SYN_r": 3.125,
            "SWEET11_export": 1.25,
            "BIOMASS_rice": 0.095,
        }
        csv_str = app.generate_fluxes_csv(fluxes)
        df = pd.read_csv(io.StringIO(csv_str))
        assert len(df) == 4
        assert list(df.columns) == ["Reaction_ID", "Flux"]
        assert df.loc[df["Reaction_ID"] == "BIOMASS_rice", "Flux"].values[0] == pytest.approx(0.095)

    def test_generate_tensor_csv_structure(self):
        """Verifies 3D phase plane tensor grid exporter produces multi-column dataframe."""
        pp = app.get_cached_phase_plane(external_nh4=0.0, grid_res=5)
        csv_str = app.generate_tensor_csv(pp)
        df = pd.read_csv(io.StringIO(csv_str))
        expected_cols = [
            "Light_pct",
            "SWEET11_export",
            "Biomass_rice",
            "Nitrogenase_ab",
            "Is_Starved",
        ]
        for col in expected_cols:
            assert col in df.columns
        assert len(df) == 5 * 5

    def test_generate_manuscript_markdown(self):
        """Verifies manuscript markdown generator handles edge results without syntax errors."""
        exp1 = app.get_cached_experiment_1(external_nh4=0.0, genotype="OsSWEET11_WT")
        exp2 = app.get_cached_hypothesis_test(50.0, external_nh4=0.0)
        exp3 = app.get_cached_bifurcation(50.0, sweet11_export=2.0)

        md = app.generate_manuscript_markdown(exp1, exp2, exp3)
        assert isinstance(md, str)
        assert "# Scientific Manuscript: Rice-Azospirillum Symbiosis" in md
        assert "Optimal SWEET11 Window" in md
        assert "Soil Ammonium Collapse & Bifurcation Analysis" in md
        assert f"[{exp2.optimal_lb:.2f}, {exp2.optimal_ub:.2f}]" in md
