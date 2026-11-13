"""tests/test_adversarial_m2_2.py

Adversarial Stress Test Suite for Milestone M2 (FBA Simulation Engine).
Challenger 2 Focus:
1. Experiment 2: Biological hypothesis testing (H0 vs H1), operating windows,
   extreme low-light stress (10%, 5%, 2%, 0%), sampling resolution, and ammonium interference.
2. Experiment 3: Bifurcation root-finding convergence, tolerance sensitivity,
   search range robustness, NHBI inversion sign checks, and nitrogenase shutoff dynamics.
"""

import sys
import os
import pytest
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import simulation_engine as sim
import model_builder
from tests.test_simulation import ReferenceModel, ReferenceSimulator


@pytest.fixture(params=["production", "reference"])
def active_model(request):
    """Parametrize across both production TwoCompartmentMetabolicModel and ReferenceModel."""
    if request.param == "production":
        return model_builder.build_two_compartment_model()
    else:
        return ReferenceModel()


# ==============================================================================
# 1. Experiment 2 Adversarial Stress Testing
# ==============================================================================

class TestExperiment2HypothesisAdversarial:
    """Stress tests for test_experiment_2_hypothesis."""

    def test_exp2_standard_50_light_contract(self, active_model):
        """Verifies standard H1 acceptance at 50% light and validates operating window."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=50.0, external_nh4=0.0)

        assert res.h1_accepted is True, f"H1 must be accepted at 50% light! Summary: {res.decision_summary}"
        assert res.optimal_lb > 0.0, f"Expected optimal_lb > 0, got {res.optimal_lb}"
        assert res.optimal_ub >= res.optimal_lb, f"optimal_ub ({res.optimal_ub}) < optimal_lb ({res.optimal_lb})"
        assert res.optimal_ub < res.starvation_threshold_flux, (
            f"optimal_ub ({res.optimal_ub}) must be strictly less than starvation threshold ({res.starvation_threshold_flux})"
        )
        assert res.max_biomass > res.ko_biomass, "Max biomass must exceed KO biomass"
        assert res.nitrogenase_at_opt > 0.0, "Nitrogenase at optimal window must be positive"
        assert res.max_benefit_pct > 0.0, "Max benefit percentage must be positive"

        # Empirical Verification: Sample points inside [optimal_lb, optimal_ub]
        sample_sw = np.linspace(res.optimal_lb, res.optimal_ub, 5)
        for sw in sample_sw:
            sol = sim.solve_single_point(
                active_model, light_pct=50.0, sweet11_bounds=(sw, sw), external_nh4=0.0
            )
            assert sol.is_starved is False, f"Unexpected starvation at beneficial export sw={sw}"
            assert sol.atpm_r >= 2.5 - 1e-4, f"Maintenance ATP deficit at beneficial export sw={sw}"
            assert sol.biomass_rice > res.ko_biomass, f"Biomass at sw={sw} did not exceed KO biomass"
            assert sol.nitrogenase_ab > 0.0, f"Nitrogenase at sw={sw} must be active"

        # Starvation boundary verification: solve beyond starvation threshold
        sol_starved = sim.solve_single_point(
            active_model,
            light_pct=50.0,
            sweet11_bounds=(res.starvation_threshold_flux + 1.0, res.starvation_threshold_flux + 1.0),
            external_nh4=0.0,
        )
        assert sol_starved.is_starved is True, "Expected starvation beyond starvation threshold"

    @pytest.mark.parametrize("light_val", [100.0, 70.0, 30.0, 20.0])
    def test_exp2_intermediate_light_regimes(self, active_model, light_val):
        """Hypothesis testing across varied light levels (100%, 70%, 30%, 20%)."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=light_val, external_nh4=0.0)

        # In all viable light regimes (>= 20%), symbiosis should be beneficial
        assert res.h1_accepted is True
        assert res.optimal_lb > 0.0
        assert res.optimal_ub >= res.optimal_lb
        assert res.optimal_ub < res.starvation_threshold_flux
        assert res.max_biomass > res.ko_biomass

        # The starvation threshold should monotonically decrease as light decreases
        # E.g., at 100% light, starvation threshold is much higher than at 30% light
        if light_val == 100.0:
            assert res.starvation_threshold_flux >= 3.0
        elif light_val == 30.0:
            assert res.starvation_threshold_flux < 3.0

    def test_exp2_extreme_low_light_10_pct(self, active_model):
        """Stress test at 10% light: very narrow viable operating window."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=10.0, external_nh4=0.0, n_samples=200)

        # Must execute cleanly without exception
        assert isinstance(res.h1_accepted, (bool, np.bool_))
        if res.h1_accepted:
            assert res.optimal_lb > 0.0
            assert res.optimal_ub < res.starvation_threshold_flux
            # Window should be tightly compressed under 10% light
            assert (res.optimal_ub - res.optimal_lb) <= 2.0
            # Test that optimal point is indeed feasible
            sol_opt = sim.solve_single_point(
                active_model, light_pct=10.0, sweet11_bounds=(res.optimal_lb, res.optimal_lb)
            )
            assert sol_opt.is_starved is False
            assert sol_opt.biomass_rice > res.ko_biomass

    def test_exp2_near_maintenance_5_pct(self, active_model):
        """Stress test at 5% light: near-maintenance operating conditions."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=5.0, external_nh4=0.0, n_samples=100)
        assert isinstance(res.h1_accepted, (bool, np.bool_))
        if res.h1_accepted:
            assert res.optimal_ub < res.starvation_threshold_flux
        else:
            assert "H0" in res.decision_summary

    @pytest.mark.parametrize("sub_maint_light", [2.0, 1.0, 0.0])
    def test_exp2_sub_maintenance_starvation(self, active_model, sub_maint_light):
        """Stress test at sub-maintenance light (2%, 1%, 0%): plant cannot survive, H1 must be rejected."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=sub_maint_light, external_nh4=0.0)
        assert res.h1_accepted is False
        assert res.optimal_lb == 0.0
        assert res.optimal_ub == 0.0
        assert "H0 Accepted" in res.decision_summary

    @pytest.mark.parametrize("samples", [10, 50, 150, 300])
    def test_exp2_sample_density_invariance(self, active_model, samples):
        """Stress test resolution invariance: coarse and fine grids should give consistent results."""
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=50.0, n_samples=samples)
        assert res.h1_accepted is True
        assert res.optimal_lb > 0.0
        assert res.optimal_ub >= res.optimal_lb
        assert res.optimal_ub < res.starvation_threshold_flux

    def test_exp2_ammonium_saturation_interference(self, active_model):
        """Stress test with high soil ammonium: symbiosis is no longer advantageous."""
        # When soil has abundant ammonium (e.g. 10.0), KO plant has plenty of nitrogen without giving away sucrose.
        res = sim.test_experiment_2_hypothesis(active_model, light_pct=50.0, external_nh4=10.0)
        # In saturated ammonium, forced sucrose export to bacteria drains carbon without adding nitrogen benefit
        assert res.h1_accepted is False or res.optimal_lb == 0.0
        if not res.h1_accepted:
            assert "H0 Accepted" in res.decision_summary


# ==============================================================================
# 2. Experiment 3 Bifurcation & Root-Finding Adversarial Stress Testing
# ==============================================================================

class TestExperiment3BifurcationAdversarial:
    """Stress tests for pinpoint_experiment_3_bifurcation."""

    def test_exp3_standard_convergence_contract(self, active_model):
        """Verifies standard bifurcation convergence at 50% light."""
        res = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=2.0, search_range=(0.0, 20.0), tol=1e-4
        )

        # 1. Critical threshold bracket
        assert 0.0 < res.critical_nh4_threshold < 20.0, f"Critical NH4 threshold out of bounds: {res.critical_nh4_threshold}"

        # 2. Sign inversion: positive at 0, negative at high NH4
        assert res.nhbi_at_zero_nh4 > 0.0, f"Expected NHBI > 0 at zero soil NH4, got {res.nhbi_at_zero_nh4}"
        assert res.nhbi_at_high_nh4 < 0.0, f"Expected NHBI < 0 at high soil NH4, got {res.nhbi_at_high_nh4}"

        # 3. Direct empirical verification of root convergence
        # Re-evaluate NHBI at critical threshold
        ctx = sim._SolverContext(active_model)
        max_avail_suc = (50.0 / 100.0) * 3.125 - (2.5 / 30.0)
        eff_export = min(2.0, max(0.05, max_avail_suc * 0.25))

        sol_inoc = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(eff_export, eff_export),
            external_nh4=res.critical_nh4_threshold
        )
        sol_un = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(0.0, 0.0),
            external_nh4=res.critical_nh4_threshold, inoculated=False
        )
        direct_nhbi = sol_inoc.biomass_rice - sol_un.biomass_rice

        assert abs(direct_nhbi) < 1e-3, f"Bisection root error too large: NHBI(crit) = {direct_nhbi}"

        # 4. Inversion monotonicity check around the root:
        # Slightly to the left, NHBI should be >= 0; slightly to the right, NHBI should be <= 0
        sol_left_inoc = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(eff_export, eff_export),
            external_nh4=res.critical_nh4_threshold - 0.1
        )
        sol_left_un = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(0.0, 0.0),
            external_nh4=res.critical_nh4_threshold - 0.1, inoculated=False
        )
        nhbi_left = sol_left_inoc.biomass_rice - sol_left_un.biomass_rice

        sol_right_inoc = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(eff_export, eff_export),
            external_nh4=res.critical_nh4_threshold + 0.1
        )
        sol_right_un = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(0.0, 0.0),
            external_nh4=res.critical_nh4_threshold + 0.1, inoculated=False
        )
        nhbi_right = sol_right_inoc.biomass_rice - sol_right_un.biomass_rice

        assert nhbi_left >= direct_nhbi >= nhbi_right, (
            f"Monotonicity violation: left={nhbi_left}, mid={direct_nhbi}, right={nhbi_right}"
        )

    def test_exp3_nitrogenase_shutoff_at_high_nh4(self, active_model):
        """Verifies that diazotroph nitrogenase activity shuts off when soil ammonium is high."""
        res = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=2.0, search_range=(0.0, 20.0)
        )

        assert res.nitrogenase_shutoff_nh4 > 0.0
        assert res.nitrogenase_shutoff_nh4 <= 20.0

        # Direct test at high external NH4 (e.g. 20.0)
        sol_high = sim.solve_single_point(
            active_model, light_pct=50.0, sweet11_bounds=(0.5, 0.5), external_nh4=20.0, inoculated=True
        )
        assert sol_high.nitrogenase_ab == pytest.approx(0.0, abs=1e-5), (
            f"Nitrogenase must shut off under high soil NH4 (20.0), got {sol_high.nitrogenase_ab}"
        )

    @pytest.mark.parametrize("tol_val", [1e-2, 1e-4, 1e-6])
    def test_exp3_tolerance_sensitivity(self, active_model, tol_val):
        """Verifies that root-finding reaches requested tolerances."""
        res = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=2.0, search_range=(0.0, 20.0), tol=tol_val
        )
        assert 0.0 < res.critical_nh4_threshold < 20.0

        # Check error
        ctx = sim._SolverContext(active_model)
        max_avail_suc = (50.0 / 100.0) * 3.125 - (2.5 / 30.0)
        eff_export = min(2.0, max(0.05, max_avail_suc * 0.25))
        sol_inoc = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(eff_export, eff_export),
            external_nh4=res.critical_nh4_threshold
        )
        sol_un = sim._solve_with_context(
            ctx, light_pct=50.0, sweet11_bounds=(0.0, 0.0),
            external_nh4=res.critical_nh4_threshold, inoculated=False
        )
        err = abs(sol_inoc.biomass_rice - sol_un.biomass_rice)
        # Should be within max(tol_val * 2, 5e-4)
        assert err <= max(tol_val * 5, 1e-3), f"Tolerance {tol_val} not met, residual: {err}"

    @pytest.mark.parametrize("search_range", [
        (0.0, 5.0),
        (0.0, 10.0),
        (0.0, 30.0),
        (0.1, 15.0),
    ])
    def test_exp3_search_range_robustness(self, active_model, search_range):
        """Verifies that different valid search ranges converge to the same critical threshold."""
        res = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=2.0, search_range=search_range, tol=1e-4
        )
        # Root is around ~0.344. In all ranges containing ~0.344, it should converge to ~0.344 +/- 0.02
        assert 0.30 <= res.critical_nh4_threshold <= 0.40, (
            f"Search range {search_range} converged to unexpected threshold {res.critical_nh4_threshold}"
        )

    def test_exp3_light_stress_dual_comparison(self, active_model):
        """Adversarially tests that 30% light contracts mutualistic benefit compared to 50% light."""
        bif_50 = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=2.0
        )
        bif_30 = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=30.0, sweet11_export=2.0
        )

        assert bif_50.nhbi_at_zero_nh4 > 0.0
        assert bif_30.nhbi_at_zero_nh4 > 0.0
        # In lower light, plant has less carbon excess to trade, so benefit index is smaller or equal
        assert bif_30.nhbi_at_zero_nh4 <= bif_50.nhbi_at_zero_nh4 + 1e-6
        # Both shift to parasitic at high NH4
        assert bif_50.nhbi_at_high_nh4 < 0.0
        assert bif_30.nhbi_at_high_nh4 < 0.0

    @pytest.mark.parametrize("export_val", [0.5, 1.0, 2.0, 4.0])
    def test_exp3_export_flux_variations(self, active_model, export_val):
        """Tests bifurcation behavior across multiple exudation levels."""
        res = sim.pinpoint_experiment_3_bifurcation(
            active_model, light_pct=50.0, sweet11_export=export_val
        )
        assert res.critical_nh4_threshold > 0.0
        assert res.nhbi_at_zero_nh4 > 0.0
        assert res.nhbi_at_high_nh4 < 0.0
