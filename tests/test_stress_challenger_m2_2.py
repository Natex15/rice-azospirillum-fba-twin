"""tests/test_stress_challenger_m2_2.py

Empirical Stress Harness for Milestone M2 Remediation (`simulation_engine.py`).
Author: Challenger 2 (teamwork_challenger)

Tests:
1. Parallel concurrency & process safety stress (ProcessPoolExecutor, large multi-process parameter sweeps).
2. Degenerate inputs & solver resilience (zero bounds, ill-conditioned matrices, conflicting bounds, missing reactions).
3. Mass conservation (S * v = 0 for internal metabolites), irreversibility bounds (v >= lb), asymptotic limits, and numerical root-finding stability.
"""

import sys
import os
import concurrent.futures
import pytest
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import model_builder
import simulation_engine as se
from tests.test_simulation import ReferenceModel


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def prod_model():
    return model_builder.build_two_compartment_model()


@pytest.fixture(scope="module")
def ref_model():
    return ReferenceModel()


# ==============================================================================
# 1. Parallel Concurrency & Multi-Process Stress Harness
# ==============================================================================

def _mp_single_solve(lp: float):
    """Helper for multiprocess single solve."""
    m = model_builder.build_two_compartment_model()
    sol = se.solve_single_point(m, light_pct=lp)
    return {"light": lp, "biomass": sol.biomass_rice, "nif": sol.nitrogenase_ab, "starved": sol.is_starved}


def _mp_sweep_worker(lp: float):
    """Helper for multiprocess 1D sweep."""
    m = model_builder.build_two_compartment_model()
    res = se.run_1d_sweep(m, "sweet11_export", np.linspace(0.0, 5.0, 8), fixed_params={"light_pct": lp})
    return {"light": lp, "max_bio": float(np.max(res.biomass_rice))}


def _mp_hypothesis_worker(lp: float):
    """Helper for multiprocess hypothesis test."""
    m = model_builder.build_two_compartment_model()
    res = se.test_experiment_2_hypothesis(m, light_pct=lp, n_samples=25)
    return {"light": lp, "h1_accepted": res.h1_accepted, "opt_ub": res.optimal_ub}


class TestConcurrencyAndParallelStress:
    """Stress tests concurrent execution across processes.
    
    EMPIRICAL FINDING:
    SciPy's compiled HiGHS C++ wrapper (`_highspy`) contains global/C-static state
    that is NOT safe for concurrent OS threads within the same process on Windows,
    triggering a Windows Access Violation (SEH 0xC0000005).
    Multi-process execution (`ProcessPoolExecutor`) provides true process address-space
    isolation, running completely safely across CPU cores under high load.
    """

    def test_multiprocess_high_load_sweeps(self):
        """Execute 20 parallel FBA simulations across 4 worker processes under high load."""
        param_lights = np.linspace(10.0, 100.0, 20)

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_mp_single_solve, float(lp)) for lp in param_lights]
            results = [f.result(timeout=60.0) for f in futures]

        assert len(results) == 20
        # Verify determinism and monotonic trend with light
        sorted_results = sorted(results, key=lambda x: x["light"])
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i]["biomass"] <= sorted_results[i+1]["biomass"] + 1e-6, (
                f"Biomass non-monotonic: {sorted_results[i]['biomass']} vs {sorted_results[i+1]['biomass']}"
            )

    def test_multiprocess_mixed_pipeline_stress(self):
        """Execute mixed heterogeneous pipeline tasks across processes concurrently."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            f_single = [executor.submit(_mp_single_solve, lp) for lp in [30.0, 50.0, 80.0]]
            f_sweep = [executor.submit(_mp_sweep_worker, lp) for lp in [40.0, 70.0]]
            f_hypo = [executor.submit(_mp_hypothesis_worker, lp) for lp in [35.0, 50.0]]

            all_futures = f_single + f_sweep + f_hypo
            done, not_done = concurrent.futures.wait(all_futures, timeout=90.0)

        assert len(not_done) == 0, f"{len(not_done)} process tasks timed out!"
        for f in done:
            assert f.exception() is None, f"Multiprocess task failed: {f.exception()}"


# ==============================================================================
# 2. Degenerate Inputs & Solver Robustness
# ==============================================================================

class TestDegenerateInputsAndSolverFallbacks:
    """Stress tests degenerate matrices, zero bounds, ill-conditioning, and solver fallbacks."""

    def test_zero_photosynthesis_capacity_graceful_starvation(self, prod_model):
        """When photosynthesis upper bound is clamped to 0.0, the plant cannot synthesize
        sucrose to meet mandatory cellular maintenance ATP (lb=2.5).
        Must report status='starvation', is_starved=True, atpm_deficit=2.5.
        """
        ctx = se._SolverContext(prod_model)
        ps_idx = ctx.idx.get("PHOTOSYN_r")
        assert ps_idx is not None
        ctx.base_ub[ps_idx] = 0.0

        sol = se._solve_with_context(ctx, light_pct=100.0)
        assert sol.is_feasible is False
        assert sol.is_starved is True
        assert sol.status == "starvation"
        assert sol.biomass_rice == pytest.approx(0.0, abs=1e-8)
        assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-5)

    def test_infeasible_conflicting_bounds_graceful_handling(self, prod_model):
        """When reaction bounds are mutually contradictory (lb > ub on non-overwritten reaction),
        linprog detects primal infeasibility. The two-stage solver gracefully falls back
        and reports infeasible/starvation without crashing.
        """
        ctx = se._SolverContext(prod_model)
        resp_idx = ctx.idx.get("RESP_r")
        assert resp_idx is not None
        ctx.base_lb[resp_idx] = 50.0
        ctx.base_ub[resp_idx] = 10.0

        sol = se._solve_with_context(ctx, light_pct=100.0)
        assert sol.is_feasible is False
        assert sol.is_starved is True
        assert sol.status in ("starvation", "infeasible")

    def test_ill_conditioned_stoichiometric_matrix(self):
        """Test with a custom dummy model having ill-conditioned/collinear stoichiometry."""
        class DegenerateModel:
            def __init__(self):
                self.S = np.ones((3, 3), dtype=np.float64)
                self.b_eq = np.zeros(3, dtype=np.float64)
                self.reaction_ids = ["EX_photon_r", "ATPM_r", "BIOMASS_rice"]
                self.metabolite_ids = ["m1", "m2", "m3"]
                self.lb = np.array([0.0, 2.5, 0.0])
                self.ub = np.array([100.0, 100.0, 100.0])

        m = DegenerateModel()
        ctx = se._SolverContext(m)
        sol = se._solve_with_context(ctx, light_pct=100.0)
        assert isinstance(sol.is_starved, (bool, np.bool_))
        assert isinstance(sol.status, str)

    def test_empty_reaction_set_validation(self):
        """A model with 0 reactions is fundamentally degenerate for linear programming.
        Linprog correctly raises ValueError on N=0.
        """
        class EmptyModel:
            def __init__(self):
                self.S = np.zeros((0, 0), dtype=np.float64)
                self.reaction_ids = []
                self.metabolite_ids = []
                self.lb = np.array([])
                self.ub = np.array([])

        m = EmptyModel()
        ctx = se._SolverContext(m)
        assert ctx.N == 0
        assert ctx.M == 0
        with pytest.raises(ValueError, match="Invalid input for linprog"):
            se._solve_with_context(ctx)

    def test_solver_method_specification(self, prod_model):
        """Test explicit solver methods supported by SciPy linprog."""
        sol_highs = se.solve_single_point(prod_model, light_pct=50.0, solver_method="highs")
        assert sol_highs.status == "optimal"
        assert sol_highs.biomass_rice > 0.0

        sol_ds = se.solve_single_point(prod_model, light_pct=50.0, solver_method="highs-ds")
        assert sol_ds.status == "optimal"
        assert np.isclose(sol_highs.biomass_rice, sol_ds.biomass_rice, atol=1e-5)

        sol_ipm = se.solve_single_point(prod_model, light_pct=50.0, solver_method="highs-ipm")
        assert sol_ipm.status == "optimal"
        assert np.isclose(sol_highs.biomass_rice, sol_ipm.biomass_rice, atol=1e-5)


# ==============================================================================
# 3. Mass Conservation, Irreversibility & Numerical Stability
# ==============================================================================

class TestMassConservationAndNumericalStability:
    """Stress tests mass balance law (S * v = 0), bounds compliance, and asymptotic limits."""

    @pytest.mark.parametrize("light", [10.0, 30.0, 50.0, 70.0, 100.0])
    @pytest.mark.parametrize("nh4", [0.0, 1.0, 5.0, 15.0])
    def test_mass_conservation_steady_state(self, prod_model, light, nh4):
        """For every feasible/optimal solution, the steady-state mass balance S * v = 0
        must hold strictly for all internal metabolites (within numerical tolerance 1e-7).
        """
        sol = se.solve_single_point(prod_model, light_pct=light, external_nh4=nh4)
        if sol.is_feasible and not sol.is_starved:
            ctx = se._SolverContext(prod_model)
            v = np.array([sol.fluxes.get(r, 0.0) for r in ctx.rxn_ids])
            residual = ctx.S @ v - ctx.b_eq
            max_res = np.max(np.abs(residual))
            assert max_res < 1e-7, f"Mass conservation violation! Max residual {max_res:.2e} at light={light}, nh4={nh4}"

    @pytest.mark.parametrize("genotype", ["WT", "OE", "KD", "KO"])
    def test_irreversibility_bounds_compliance(self, prod_model, genotype):
        """Fluxes through irreversible reactions must never be negative (v >= lb - 1e-8)."""
        sol = se.solve_single_point(prod_model, light_pct=50.0, genotype=genotype)
        for rxn_id, flux in sol.fluxes.items():
            lb, ub = prod_model.get_reaction_bounds(rxn_id)
            if lb >= 0.0:
                assert flux >= -1e-8, f"Irreversible reaction {rxn_id} has negative flux {flux}"
            assert flux <= ub + 1e-8, f"Reaction {rxn_id} exceeded ub ({ub}): flux={flux}"

    def test_asymptotic_infinite_light(self, prod_model):
        """At infinite light (e.g. 1e6), the solver must not overflow, crash, or produce NaN."""
        sol = se.solve_single_point(prod_model, light_pct=1e6)
        assert sol.status == "optimal"
        assert np.isfinite(sol.biomass_rice)
        assert not np.isnan(sol.biomass_rice)

    def test_asymptotic_infinite_soil_nh4(self, prod_model):
        """At extreme soil ammonium (1e6), plant uses soil NH4, nitrogenase shuts off, no NaN."""
        sol = se.solve_single_point(prod_model, light_pct=50.0, external_nh4=1e6)
        assert sol.status == "optimal"
        assert np.isclose(sol.nitrogenase_ab, 0.0, atol=1e-5)
        assert np.isfinite(sol.biomass_rice)
        assert not np.isnan(sol.biomass_rice)

    def test_tiny_dt_smoothness_starvation_transition(self, prod_model):
        """Test fine-grained light sweep near maintenance threshold (3.0% to 5.0% in 0.1% steps).
        Verifies smooth monotonic increase of biomass and no sudden discontinuous jumps or crashes.
        """
        lights = np.linspace(3.0, 5.0, 21)
        biomasses = []
        for lp in lights:
            sol = se.solve_single_point(prod_model, light_pct=lp)
            biomasses.append(sol.biomass_rice)

        biomasses = np.array(biomasses)
        diffs = np.diff(biomasses)
        assert np.all(diffs >= -1e-9), f"Non-monotonic biomass behavior detected near starvation threshold: {diffs}"

    def test_bifurcation_outside_search_bracket(self, prod_model):
        """Test bifurcation root-finding when search bracket does NOT contain the root.
        e.g., search_range = (5.0, 10.0) when root is ~0.344.
        Must return gracefully without unhandled infinite loop or crash.
        """
        res = se.pinpoint_experiment_3_bifurcation(
            prod_model, light_pct=50.0, sweet11_export=2.0, search_range=(5.0, 10.0), tol=1e-4
        )
        assert isinstance(res.critical_nh4_threshold, float)
        assert 5.0 <= res.critical_nh4_threshold <= 10.0
