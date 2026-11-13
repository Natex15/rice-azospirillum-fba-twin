"""tests/test_adversarial_challenger.py

Adversarial Stress Test Suite for Milestone M2 (FBA Simulation Engine).
Empirically stress-tests numerical limits, edge cases, tensor shapes, and invariants.
"""

import sys
import numpy as np
import pytest

import model_builder
import simulation_engine as se


@pytest.fixture
def model():
    return model_builder.build_two_compartment_model()


class TestAdversarialNumericalBounds:
    def test_zero_light_starvation(self, model):
        """Test light_pct = 0.0 enforces starvation with deficit == 2.5."""
        sol = se.solve_single_point(model, light_pct=0.0)
        assert sol.status == "starvation", f"Expected 'starvation', got {sol.status}"
        assert sol.is_starved is True, f"Expected is_starved=True, got {sol.is_starved}"
        assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7), f"Expected 0 biomass, got {sol.biomass_rice}"
        assert sol.starvation_deficit == pytest.approx(2.5, abs=1e-4), f"Expected deficit 2.5, got {sol.starvation_deficit}"
        assert sol.atpm_deficit == pytest.approx(2.5, abs=1e-4), f"Expected atpm_deficit 2.5, got {sol.atpm_deficit}"

    def test_negative_light_graceful_handling(self, model):
        """Test extreme negative light (e.g. -50.0) clamps to 0 and starves gracefully."""
        sol = se.solve_single_point(model, light_pct=-50.0)
        assert sol.status == "starvation", f"Expected 'starvation', got {sol.status}"
        assert sol.is_starved is True
        assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)
        assert sol.starvation_deficit == pytest.approx(2.5, abs=1e-4)

    def test_huge_light_saturation(self, model):
        """Test huge light (e.g. 500.0) solves to optimal without numerical explosion."""
        sol = se.solve_single_point(model, light_pct=500.0)
        assert sol.status == "optimal"
        assert sol.is_starved is False
        assert sol.biomass_rice > 0.0
        assert np.isfinite(sol.biomass_rice)

    @pytest.mark.parametrize("extreme_sweet11", [50.0, 100.0, 500.0, 10000.0])
    def test_extreme_sweet11_forcing_starvation(self, model, extreme_sweet11):
        """Test forcing SWEET11 export beyond photosynthetic capacity starves gracefully."""
        sol = se.solve_single_point(model, light_pct=100.0, sweet11_bounds=(extreme_sweet11, extreme_sweet11))
        assert sol.status == "starvation"
        assert sol.is_starved is True
        assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)
        assert sol.starvation_deficit > 0.0

    @pytest.mark.parametrize("light", [0.0, 10.0, 30.0, 50.0, 70.0, 100.0, 150.0])
    def test_uninoculated_mode_strictly_zero_bacterial_fluxes(self, model, light):
        """Test that in uninoculated mode, all bacterial fluxes are strictly zero."""
        sol = se.solve_single_point(model, light_pct=light, inoculated=False)
        bacterial_reactions = [
            "SUC_uptake_ab",
            "RESP_ab",
            "ATPM_ab",
            "NITROGENASE_ab",
            "NH4_excretion_ab",
            "NH4_uptake_ab",
            "BIOMASS_ab",
            "SK_biomass_ab",
        ]
        for rxn in bacterial_reactions:
            if rxn in sol.fluxes:
                assert sol.fluxes[rxn] == pytest.approx(0.0, abs=1e-9), (
                    f"Bacterial reaction {rxn} non-zero ({sol.fluxes[rxn]}) at light={light} in uninoculated mode"
                )
        assert sol.sweet11_export == pytest.approx(0.0, abs=1e-9)
        assert sol.biomass_ab == pytest.approx(0.0, abs=1e-9)
        assert sol.nitrogenase_ab == pytest.approx(0.0, abs=1e-9)


class TestAdversarialTensorShapesAndIndexing:
    def test_non_square_2d_phase_plane(self, model):
        """Test generate_2d_phase_plane with non-square grid dimensions Nx=7, Ny=3."""
        nx = 7
        ny = 3
        res = se.generate_2d_phase_plane(
            model,
            param_x=("light_pct", 10.0, 100.0, nx),
            param_y=("sweet11_export", 0.0, 5.0, ny),
            compute_nhbi=True,
        )
        assert res.X.shape == (ny, nx), f"X.shape {res.X.shape} != ({ny}, {nx})"
        assert res.Y.shape == (ny, nx), f"Y.shape {res.Y.shape} != ({ny}, {nx})"
        assert res.Z_biomass_rice.shape == (ny, nx), f"Z_biomass shape {res.Z_biomass_rice.shape} != ({ny}, {nx})"
        assert res.Z_plant.shape == (ny, nx)
        assert res.Z_nitrogenase_ab.shape == (ny, nx)
        assert res.Z_nhbi.shape == (ny, nx)
        assert res.starvation_mask.shape == (ny, nx)
        assert len(res.pareto_ridge_x) == nx, f"ridge_x len {len(res.pareto_ridge_x)} != {nx}"
        assert len(res.pareto_ridge_y) == nx, f"ridge_y len {len(res.pareto_ridge_y)} != {nx}"

    def test_column_zero_parity_between_2d_and_1d(self, model):
        """Verify column 0 of 2D phase plane exactly matches 1D sweep at y=y_min."""
        nx = 6
        ny = 4
        y_min = 0.0
        x_min, x_max = 20.0, 100.0

        res_2d = se.generate_2d_phase_plane(
            model,
            param_x=("light_pct", x_min, x_max, nx),
            param_y=("sweet11_export", y_min, 10.0, ny),
            compute_nhbi=False,
        )

        sweep_1d = se.run_1d_sweep(
            model,
            param_name="light_pct",
            param_values=np.linspace(x_min, x_max, nx),
            fixed_params={"sweet11_bounds": (y_min, y_min)},
        )

        # In indexing='xy', row 0 corresponds to y_min
        for i in range(nx):
            val_2d = res_2d.Z_biomass_rice[0, i]
            val_1d = sweep_1d.biomass_rice[i]
            assert val_2d == pytest.approx(val_1d, rel=1e-4, abs=1e-6), (
                f"Mismatch at col {i}: 2D={val_2d}, 1D={val_1d}"
            )

    def test_starvation_cliff_properties(self, model):
        """Test starvation cliff coordinates are valid finite arrays."""
        res = se.generate_2d_phase_plane(
            model,
            param_x=("light_pct", 10.0, 100.0, 10),
            param_y=("sweet11_export", 0.0, 10.0, 10),
        )
        assert len(res.cliff_x) >= 2
        assert len(res.cliff_y) >= 2
        assert len(res.cliff_x) == len(res.cliff_y)
        assert np.all(np.isfinite(res.cliff_x))
        assert np.all(np.isfinite(res.cliff_y))

    def test_non_cubic_3d_tensor_field(self, model):
        """Test generate_3d_tensor_field with non-cubic dimensions (5, 3, 2)."""
        nx, ny, nz = 5, 3, 2
        t_field = se.generate_3d_tensor_field(
            model,
            light_range=(20.0, 80.0, nx),
            sweet11_range=(0.0, 6.0, ny),
            nh4_range=(0.0, 4.0, nz),
        )
        assert t_field["T_biomass"].shape == (nx, ny, nz)
        assert t_field["T_nitrogenase"].shape == (nx, ny, nz)
        assert t_field["T_nhbi"].shape == (nx, ny, nz)
        assert t_field["M_starvation"].shape == (nx, ny, nz)
        assert np.all(np.isfinite(t_field["T_biomass"]))
        assert np.all(np.isfinite(t_field["T_nitrogenase"]))


class TestAdversarialModelImmutability:
    def test_model_not_mutated_after_extreme_solves(self, model):
        """Verify solving with extreme parameters does not mutate model bounds."""
        lb_orig = model.lb.copy()
        ub_orig = model.ub.copy()

        se.solve_single_point(model, light_pct=-100.0, sweet11_bounds=(999.0, 999.0))
        se.solve_single_point(model, light_pct=1000.0, external_nh4=500.0)
        se.solve_uninoculated_baseline(model, light_pct=0.0)
        se.generate_2d_phase_plane(model, ("light_pct", 0, 50, 3), ("sweet11_export", 0, 20, 3))
        se.test_experiment_2_hypothesis(model, light_pct=50.0, n_samples=10)
        se.pinpoint_experiment_3_bifurcation(model, light_pct=50.0)

        np.testing.assert_array_equal(model.lb, lb_orig, err_msg="Model lower bounds were mutated!")
        np.testing.assert_array_equal(model.ub, ub_orig, err_msg="Model upper bounds were mutated!")


class TestAdversarialEdgeConditions:
    def test_extreme_negative_soil_nh4(self, model):
        """Negative soil ammonium should clamp to 0.0."""
        sol = se.solve_single_point(model, external_nh4=-10.0)
        assert sol.ex_nh4_soil == pytest.approx(0.0, abs=1e-7)

    def test_zero_light_and_forced_export(self, model):
        """Zero light with positive SWEET11 export should yield starvation and deficit."""
        sol = se.solve_single_point(model, light_pct=0.0, sweet11_bounds=(2.0, 2.0))
        assert sol.is_starved is True
        assert sol.status == "starvation"
        assert sol.biomass_rice == pytest.approx(0.0, abs=1e-7)

    def test_experiment_1_extreme_monotone(self, model):
        """Verify experiment 1 seasonal monotonicity under unfertilized soil."""
        res = se.simulate_experiment_1_seasonal(model, external_nh4=0.0)
        b_dry = res.regimes["100% Peak Dry Season Control"].biomass_rice
        b_mon = res.regimes["70% Intermittent Monsoon Cloud Cover"].biomass_rice
        b_rai = res.regimes["50% Extended Rainy Season Overcast"].biomass_rice
        b_typ = res.regimes["30% Active Typhoon Passage"].biomass_rice

        assert b_dry > b_mon > b_rai > b_typ >= 0.0

    def test_bifurcation_at_low_light(self, model):
        """Test bifurcation pinpointing when light is low."""
        bifurc = se.pinpoint_experiment_3_bifurcation(model, light_pct=30.0)
        assert np.isfinite(bifurc.critical_nh4_threshold)
        assert bifurc.nhbi_at_zero_nh4 > 0.0
        assert bifurc.nhbi_at_high_nh4 < 0.0

    def test_hypothesis_testing_under_darkness(self, model):
        """At 0% light, host cannot support mutualism; H1 must be rejected (H0 accepted)."""
        hyp = se.test_experiment_2_hypothesis(model, light_pct=0.0, n_samples=20)
        assert hyp.h1_accepted is False
        assert hyp.optimal_lb == 0.0
        assert hyp.optimal_ub == 0.0

    def test_unknown_genotype_graceful_fallback(self, model):
        """Unknown genotype string should not crash, but fallback gracefully."""
        sol = se.solve_single_point(model, genotype="UNKNOWN_CUSTOM_MUTANT")
        assert sol.status == "optimal"
        assert sol.biomass_rice > 0.0

    def test_downstream_property_contract_aliases(self, model):
        """Verify all dataclass property aliases for visualizer_3d.py."""
        phase = se.generate_2d_phase_plane(
            model,
            param_x=("light_pct", 20.0, 80.0, 4),
            param_y=("sweet11_export", 0.0, 4.0, 3),
        )
        # Test PhasePlane2DResult properties
        assert phase.X_light is phase.X
        assert phase.Y_sweet11 is phase.Y
        assert phase.Z_rice_biomass is phase.Z_biomass_rice
        assert phase.Z_plant is phase.Z_biomass_rice
        assert phase.Z_nitrogenase is phase.Z_nitrogenase_ab
        assert phase.Z_bact is phase.Z_nitrogenase_ab
        assert phase.Starvation_Mask is phase.starvation_mask
        assert len(phase.light_levels) == 4
        assert len(phase.sweet11_fluxes) == 3

        # Test AmmoniumBifurcationResult properties
        bif = se.pinpoint_experiment_3_bifurcation(model, light_pct=50.0)
        assert bif.X is bif.X_nh4
        assert bif.Y is bif.Y_sweet11
        assert bif.Z_sucrose_consumed is bif.Z_nitrogenase
        assert len(bif.nh4_levels) == 10
        assert len(bif.sweet11_fluxes) == 10
