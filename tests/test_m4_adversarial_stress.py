"""tests/test_m4_adversarial_stress.py.

Empirical Adversarial Stress Testing Suite for Milestone M4:
- Exporter Pipeline Stress Testing (generate_fluxes_csv, generate_tensor_csv, generate_manuscript_markdown)
- Visualizer Integration Stress Testing (Three.js WebGL, Plotly 3D anatomical scene, Pareto surface, Ammonium collapse surface, Seasonal trajectory)
- Performance, Concurrency, and Throughput Benchmarks under heavy load
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import io
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import app
import model_builder
import simulation_engine
import visualizer_3d


# ==============================================================================
# Test Fixtures & Synthetic Generators
# ==============================================================================


@pytest.fixture(scope="module")
def default_model():
    """Builds base two-compartment model once for test module."""
    return model_builder.build_two_compartment_model()


@pytest.fixture(scope="module")
def sample_exp1_result(default_model):
    """Generates standard Experiment 1 seasonal result."""
    return simulation_engine.simulate_experiment_1_seasonal(
        model=default_model, external_nh4=0.0, genotype="OsSWEET11_WT", continuous_points=25
    )


@pytest.fixture(scope="module")
def sample_phase_plane(default_model):
    """Generates standard 2D phenotypic phase plane result."""
    return simulation_engine.simulate_experiment_2_phase_plane(
        model=default_model, external_nh4=0.0, grid_resolution=15
    )


@pytest.fixture(scope="module")
def sample_hypothesis_result(default_model):
    """Generates standard hypothesis testing result."""
    return simulation_engine.test_experiment_2_hypothesis(
        model=default_model, light_pct=50.0, external_nh4=0.0, n_samples=50
    )


@pytest.fixture(scope="module")
def sample_bifurcation_result(default_model):
    """Generates standard soil ammonium bifurcation result."""
    return simulation_engine.pinpoint_experiment_3_bifurcation(
        model=default_model, light_pct=50.0, sweet11_export=2.0, search_range=(0.0, 20.0)
    )


# ==============================================================================
# SECTION 1: Exporter Pipeline Adversarial Stress Testing
# ==============================================================================


class TestExporterPipelineStress:
    """Adversarial stress testing on CSV and Markdown exporters."""

    def test_fluxes_csv_none_input_vulnerability(self):
        """Pass None as the argument to generate_fluxes_csv."""
        with pytest.raises(AttributeError):
            app.generate_fluxes_csv(None)  # None has no .items()

    def test_fluxes_csv_empty_dict(self):
        """Pass empty dictionary to generate_fluxes_csv."""
        csv_out = app.generate_fluxes_csv({})
        assert isinstance(csv_out, str)
        # Should contain headers
        lines = [line.strip() for line in csv_out.strip().splitlines() if line.strip()]
        assert len(lines) == 1
        assert "Reaction_ID,Flux" in lines[0]

        # Verify roundtrip with pandas
        df_roundtrip = pd.read_csv(io.StringIO(csv_out))
        assert list(df_roundtrip.columns) == ["Reaction_ID", "Flux"]
        assert len(df_roundtrip) == 0

    def test_fluxes_csv_none_values(self):
        """Pass None values and mixed None keys/values."""
        fluxes_with_none = {
            "EX_photon_r": None,
            "PHOTOSYN_r": 25.4,
            "SUC_SYN_r": None,
            "BIOMASS_rice": 0.12345,
        }
        csv_out = app.generate_fluxes_csv(fluxes_with_none)
        assert isinstance(csv_out, str)
        df_roundtrip = pd.read_csv(io.StringIO(csv_out))
        assert len(df_roundtrip) == 4
        assert pd.isna(df_roundtrip.loc[df_roundtrip["Reaction_ID"] == "EX_photon_r", "Flux"].values[0])
        assert df_roundtrip.loc[df_roundtrip["Reaction_ID"] == "PHOTOSYN_r", "Flux"].values[0] == pytest.approx(25.4)

    def test_fluxes_csv_extreme_numeric_values(self):
        """Pass extreme float values (subnormals, max float, inf, nan, negatives)."""
        extreme_fluxes = {
            "MAX_FLOAT": 1e308,
            "MIN_POSITIVE": 1e-308,
            "SUBNORMAL": 5e-324,
            "INF": float("inf"),
            "NEG_INF": float("-inf"),
            "NAN": float("nan"),
            "ZERO": 0.0,
            "NEG_ZERO": -0.0,
            "LARGE_NEG": -1e300,
        }
        csv_out = app.generate_fluxes_csv(extreme_fluxes)
        assert isinstance(csv_out, str)
        df_roundtrip = pd.read_csv(io.StringIO(csv_out))
        assert len(df_roundtrip) == len(extreme_fluxes)
        assert "MAX_FLOAT" in df_roundtrip["Reaction_ID"].values
        assert "INF" in df_roundtrip["Reaction_ID"].values

    def test_fluxes_csv_special_characters_and_quoting(self):
        """Pass reaction IDs with commas, quotes, newlines, and unicode."""
        special_fluxes = {
            "EX_α_glucose[c]": 1.5,
            'Rxn,"with,commas"': 2.3,
            'Rxn"With"Quotes': 3.1,
            "Rxn\tWithTab": 4.0,
            "🌾Rice_Biomass": 0.12,
        }
        csv_out = app.generate_fluxes_csv(special_fluxes)
        assert isinstance(csv_out, str)
        df_roundtrip = pd.read_csv(io.StringIO(csv_out))
        assert len(df_roundtrip) == len(special_fluxes)
        # Check that quoting properly protected commas
        assert 'Rxn,"with,commas"' in df_roundtrip["Reaction_ID"].values
        assert "🌾Rice_Biomass" in df_roundtrip["Reaction_ID"].values

    def test_fluxes_csv_massive_scale(self):
        """Pass massive dictionary (10,000 synthetic reactions) to evaluate memory and speed."""
        large_fluxes = {f"R_{i:06d}": float(i) * 0.01 for i in range(10000)}
        t0 = time.perf_counter()
        csv_out = app.generate_fluxes_csv(large_fluxes)
        elapsed = time.perf_counter() - t0
        assert isinstance(csv_out, str)
        assert elapsed < 1.0  # Must generate 10k rows in under 1 second
        assert len(csv_out.splitlines()) == 10001

    def test_tensor_csv_1d_input_vulnerability(self):
        """Pass a 1D sweep result to generate_tensor_csv (contract specifies 2D tensor)."""
        @dataclass
        class MockSweep1D:
            X: np.ndarray = field(default_factory=lambda: np.array([10.0, 50.0, 100.0]))
            Y: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

        # X is 1D so X.shape[1] raises IndexError: tuple index out of range
        with pytest.raises(IndexError):
            app.generate_tensor_csv(MockSweep1D())

    def test_tensor_csv_standard_result(self, sample_phase_plane):
        """Verify tensor CSV generation against standard PhasePlane2DResult."""
        csv_out = app.generate_tensor_csv(sample_phase_plane)
        assert isinstance(csv_out, str)
        df = pd.read_csv(io.StringIO(csv_out))
        required_cols = ["Light_pct", "SWEET11_export", "Biomass_rice", "Nitrogenase_ab", "Is_Starved"]
        for col in required_cols:
            assert col in df.columns
        expected_rows = sample_phase_plane.X.shape[0] * sample_phase_plane.X.shape[1]
        assert len(df) == expected_rows

    def test_tensor_csv_sparse_and_1x1_grid(self):
        """Pass minimal 1x1 grid and 2x2 grid."""
        @dataclass
        class MockPhase1x1:
            X: np.ndarray = field(default_factory=lambda: np.array([[50.0]]))
            Y: np.ndarray = field(default_factory=lambda: np.array([[2.0]]))
            Z_biomass_rice: np.ndarray = field(default_factory=lambda: np.array([[0.08]]))
            Z_nitrogenase_ab: np.ndarray = field(default_factory=lambda: np.array([[1.2]]))
            starvation_mask: np.ndarray = field(default_factory=lambda: np.array([[False]]))

        csv_out = app.generate_tensor_csv(MockPhase1x1())
        df = pd.read_csv(io.StringIO(csv_out))
        assert len(df) == 1
        assert df["Light_pct"].iloc[0] == 50.0
        assert df["SWEET11_export"].iloc[0] == 2.0
        assert df["Biomass_rice"].iloc[0] == pytest.approx(0.08)
        assert df["Nitrogenase_ab"].iloc[0] == pytest.approx(1.2)
        assert df["Is_Starved"].iloc[0] is False or df["Is_Starved"].iloc[0] == 0

    def test_tensor_csv_all_starved_grid(self):
        """Test with completely starved 10x10 grid (Z_bio=0, mask=True)."""
        shape = (10, 10)
        x_grid, y_grid = np.meshgrid(np.linspace(10, 30, 10), np.linspace(5, 15, 10))
        @dataclass
        class MockAllStarved:
            X: np.ndarray = field(default_factory=lambda: x_grid)
            Y: np.ndarray = field(default_factory=lambda: y_grid)
            Z_biomass_rice: np.ndarray = field(default_factory=lambda: np.zeros(shape))
            Z_nitrogenase_ab: np.ndarray = field(default_factory=lambda: np.zeros(shape))
            starvation_mask: np.ndarray = field(default_factory=lambda: np.ones(shape, dtype=bool))

        csv_out = app.generate_tensor_csv(MockAllStarved())
        df = pd.read_csv(io.StringIO(csv_out))
        assert len(df) == 100
        assert (df["Biomass_rice"] == 0.0).all()
        assert (df["Nitrogenase_ab"] == 0.0).all()
        assert (df["Is_Starved"] == True).all()

    def test_tensor_csv_extreme_and_nan_values(self):
        """Pass arrays containing NaNs, Infinities, and extreme coordinate bounds."""
        X = np.array([[0.0, 1e6], [1e6, 1e9]])
        Y = np.array([[-100.0, 0.0], [50.0, 1e4]])
        Z_bio = np.array([[np.nan, 0.5], [float("inf"), -1.0]])
        Z_nif = np.array([[float("-inf"), 0.0], [np.nan, 1e5]])
        mask = np.array([[True, False], [False, True]])

        @dataclass
        class MockExtreme:
            X: np.ndarray = field(default_factory=lambda: X)
            Y: np.ndarray = field(default_factory=lambda: Y)
            Z_biomass_rice: np.ndarray = field(default_factory=lambda: Z_bio)
            Z_nitrogenase_ab: np.ndarray = field(default_factory=lambda: Z_nif)
            starvation_mask: np.ndarray = field(default_factory=lambda: mask)

        csv_out = app.generate_tensor_csv(MockExtreme())
        assert isinstance(csv_out, str)
        df = pd.read_csv(io.StringIO(csv_out))
        assert len(df) == 4

    def test_tensor_csv_fallback_attribute_names(self):
        """Verify compatibility with alternate attribute naming (e.g. Z_rice_biomass, Starvation_Mask)."""
        shape = (3, 3)
        x_grid, y_grid = np.meshgrid(np.linspace(10, 50, 3), np.linspace(1, 5, 3))
        @dataclass
        class MockAlternateAttrs:
            X: np.ndarray = field(default_factory=lambda: x_grid)
            Y: np.ndarray = field(default_factory=lambda: y_grid)
            Z_rice_biomass: np.ndarray = field(default_factory=lambda: np.full(shape, 0.09))
            Z_nitrogenase: np.ndarray = field(default_factory=lambda: np.full(shape, 0.8))
            Starvation_Mask: np.ndarray = field(default_factory=lambda: np.zeros(shape, dtype=bool))

        csv_out = app.generate_tensor_csv(MockAlternateAttrs())
        df = pd.read_csv(io.StringIO(csv_out))
        assert len(df) == 9
        assert df["Biomass_rice"].iloc[0] == pytest.approx(0.09)
        assert df["Nitrogenase_ab"].iloc[0] == pytest.approx(0.8)

    def test_manuscript_markdown_standard(
        self, sample_exp1_result, sample_hypothesis_result, sample_bifurcation_result
    ):
        """Verify standard manuscript markdown generation."""
        md = app.generate_manuscript_markdown(
            sample_exp1_result, sample_hypothesis_result, sample_bifurcation_result
        )
        assert isinstance(md, str)
        # Check required sections
        assert "# Scientific Manuscript: Rice-Azospirillum Symbiosis" in md
        assert "## Executive Summary" in md
        assert "## Methods" in md
        assert "## Results & Hypothesis Testing" in md
        assert "### Experiment 1: Philippine Seasonal Light Regimes" in md
        assert "### Experiment 2: OsSWEET11 Constraints & Phenotypic Phase Plane" in md
        assert "### Experiment 3: Soil Ammonium Collapse & Bifurcation Analysis" in md
        assert "## Discussion & Agronomic Implications" in md
        assert "## References" in md

        # Verify mathematical rendering tags
        assert "$v_{ATPM\\_r} \\ge 2.5" in md or "$v_{ATPM_r} \\ge 2.5" in md or "ATPM" in md
        assert "NHBI" in md
        assert "H_1" in md

    def test_manuscript_markdown_with_empty_or_mock_objects(self):
        """Pass empty objects or objects with None attributes to test fallback resilience."""
        class EmptyMock:
            pass

        md_fallback = app.generate_manuscript_markdown(EmptyMock(), EmptyMock(), EmptyMock())
        assert isinstance(md_fallback, str)
        assert "# Scientific Manuscript" in md_fallback
        assert "Optimal SWEET11 Window" in md_fallback
        assert "Critical Threshold" in md_fallback

    def test_manuscript_markdown_extreme_values(self):
        """Pass extreme values in hypothesis and bifurcation results."""
        @dataclass
        class MockExp2Extreme:
            h1_accepted: bool = False
            optimal_lb: float = 0.0
            optimal_ub: float = 999.999
            max_benefit_pct: float = -45.5
            starvation_threshold_flux: float = 0.001

        @dataclass
        class MockExp3Extreme:
            critical_nh4_threshold: float = 0.0

        md = app.generate_manuscript_markdown(None, MockExp2Extreme(), MockExp3Extreme())
        # The template renders LaTeX math delimiters "$H_1$ Accepted = False"
        assert "$H_1$ Accepted = False" in md
        assert "0.00" in md
        assert "-45.5%" in md

    def test_manuscript_markdown_none_attributes_vulnerability(self):
        """Empirically test vulnerability when experiment objects have explicit None attributes."""
        @dataclass
        class MockExp2WithNone:
            optimal_lb: Optional[float] = None
            optimal_ub: Optional[float] = None

        # When optimal_lb is explicitly None, getattr(exp2, "optimal_lb", 0.10) returns None.
        # Then format string {opt_lb:.2f} raises TypeError: unsupported format string passed to NoneType.__format__
        with pytest.raises(TypeError) as excinfo:
            app.generate_manuscript_markdown(None, MockExp2WithNone(), None)
        assert "unsupported format string" in str(excinfo.value) or "NoneType" in str(excinfo.value)


# ==============================================================================
# SECTION 2: Visualizer Integration Adversarial Stress Testing
# ==============================================================================


class TestVisualizerIntegrationStress:
    """Adversarial stress testing on Three.js WebGL and Plotly 3D visualizers."""

    def test_threejs_zero_and_extreme_light(self):
        """Verify Three.js HTML generation under zero and extreme solar radiation."""
        # Zero light (complete darkness / night)
        html_zero = visualizer_3d.render_threejs_rhizosphere(
            light_pct=0.0,
            sweet11_flux=0.0,
            nh4_flux=0.0,
            is_starved=True,
            is_parasitic=False,
            height=500,
        )
        assert isinstance(html_zero, str)
        assert "<!DOCTYPE html>" in html_zero
        assert "three.min.js" in html_zero
        assert "OrbitControls.js" in html_zero
        assert "LETHAL CARBON STARVATION" in html_zero
        assert "0.0%" in html_zero

        # Extreme high light (300% photon influx)
        html_extreme = visualizer_3d.render_threejs_rhizosphere(
            light_pct=300.0,
            sweet11_flux=20.0,
            nh4_flux=15.0,
            is_starved=False,
            is_parasitic=False,
            height=700,
        )
        assert isinstance(html_extreme, str)
        assert "300.0%" in html_extreme
        assert "MUTUALISTIC SYMBIOSIS" in html_extreme

    def test_threejs_negative_fluxes_and_sanitization(self):
        """Pass negative fluxes to Three.js renderer (must be clamped safely to >=0)."""
        html_neg = visualizer_3d.render_threejs_rhizosphere(
            light_pct=-50.0,
            sweet11_flux=-10.0,
            nh4_flux=-5.0,
            is_starved=False,
            is_parasitic=True,
        )
        assert isinstance(html_neg, str)
        assert "PARASITIC CARBON DRAIN" in html_neg
        # Confirm that JavaScript code does not contain negative particle counts or NaN
        assert "NaN" not in html_neg
        assert "0.0%" in html_neg

    def test_threejs_all_state_transitions(self):
        """Verify HTML telemetry badge and color assignments across all ecological states."""
        # 1. Starved state
        h_starved = visualizer_3d.render_threejs_rhizosphere(50.0, 2.0, 1.0, is_starved=True, is_parasitic=False)
        assert "badge-starved" in h_starved
        assert "#582d0e" in h_starved  # Necrotic canopy

        # 2. Parasitic state
        h_parasitic = visualizer_3d.render_threejs_rhizosphere(50.0, 2.0, 0.0, is_starved=False, is_parasitic=True)
        assert "badge-parasitic" in h_parasitic
        assert "#ef4444" in h_parasitic  # Crimson parasite

        # 3. Mutualistic state
        h_mutual = visualizer_3d.render_threejs_rhizosphere(100.0, 2.0, 1.5, is_starved=False, is_parasitic=False)
        assert "badge-mutualist" in h_mutual
        assert "#22c55e" in h_mutual  # Green canopy

    def test_plotly_anatomical_scene_empty_and_none_fluxes(self):
        """Test plot_3d_anatomical_scene with empty dict and None."""
        fig_empty = visualizer_3d.plot_3d_anatomical_scene(fluxes={})
        assert fig_empty is not None
        assert hasattr(fig_empty, "data")
        assert len(fig_empty.data) > 0

        fig_none = visualizer_3d.plot_3d_anatomical_scene(fluxes=None)
        assert fig_none is not None
        assert len(fig_none.data) > 0

    def test_plotly_anatomical_scene_cutaway_toggle(self):
        """Verify soil cylinder mesh generation with cutaway ON (90 deg) and OFF (full 360 deg)."""
        fig_cutaway = visualizer_3d.plot_3d_anatomical_scene(fluxes={}, show_soil_cutaway=True)
        fig_full = visualizer_3d.plot_3d_anatomical_scene(fluxes={}, show_soil_cutaway=False)

        assert fig_cutaway is not None and fig_full is not None
        # Soil mesh is trace 0
        mesh_cutaway = fig_cutaway.data[0]
        mesh_full = fig_full.data[0]
        assert mesh_cutaway.type == "mesh3d"
        assert mesh_full.type == "mesh3d"
        # Cutaway has additional radial wall triangles, so index counts differ
        assert len(mesh_cutaway.i) != len(mesh_full.i)

    def test_plotly_anatomical_scene_extreme_fluxes(self):
        """Pass extremely large and negative fluxes to 3D anatomical scene."""
        extreme_fluxes = {
            "EX_photon_r": 10000.0,
            "SWEET11_export": 5000.0,
            "NITROGENASE_ab": 2500.0,
            "NH4_excretion_ab": 5000.0,
            "BIOMASS_rice": 100.0,
            "ATPM_r": 2.50,
        }
        fig = visualizer_3d.plot_3d_anatomical_scene(
            fluxes=extreme_fluxes, is_starved=False, is_parasitic=False
        )
        assert fig is not None
        # Verify cones exist and have non-zero coordinates
        cone_traces = [t for t in fig.data if t.type == "cone"]
        assert len(cone_traces) == 3  # Photon, SWEET11, NH4 cones

    def test_plotly_pareto_surface_none_and_degenerate(self):
        """Pass None and malformed phase results to plot_3d_pareto_surface."""
        # 1. None input
        fig_none = visualizer_3d.plot_3d_pareto_surface(phase_result=None)
        assert fig_none is not None
        assert "No Phase Plane Data" in str(fig_none.layout.annotations[0].text)

        # 2. Incomplete arrays
        @dataclass
        class MockIncomplete:
            X: np.ndarray = field(default_factory=lambda: np.array([]))
            Y: np.ndarray = field(default_factory=lambda: np.array([]))
            Z_biomass_rice: Optional[np.ndarray] = None

        fig_incomp = visualizer_3d.plot_3d_pareto_surface(MockIncomplete())
        assert fig_incomp is not None

    def test_plotly_pareto_surface_all_genotypes(self, sample_phase_plane):
        """Verify dynamic genotype marker highlighting across all presets and custom names."""
        genotypes_to_test = [
            "OsSWEET11_WT",
            "OsSWEET11_OE",
            "OsSWEET11_Knockdown",
            "Ossweet11_KO",
            "WT",
            "OE",
            "KO",
            "Custom_Genotype_X",
            None,
        ]
        for geno in genotypes_to_test:
            fig = visualizer_3d.plot_3d_pareto_surface(sample_phase_plane, current_genotype=geno)
            assert fig is not None
            assert len(fig.data) >= 5  # Surface, Ridge, Cliff, + 4 Genotype markers

    def test_plotly_ammonium_collapse_surface_boundaries(self, sample_bifurcation_result):
        """Verify 3D ammonium collapse surface with normal and boundary datasets."""
        # Normal
        fig_norm = visualizer_3d.plot_3d_ammonium_collapse_surface(sample_bifurcation_result)
        assert fig_norm is not None
        # Verify surface and Z=0 neutral plane exist
        surface_traces = [t for t in fig_norm.data if t.type == "surface"]
        assert len(surface_traces) >= 2  # NHBI surface + Z=0 neutral plane

        # Boundary: None input
        fig_none = visualizer_3d.plot_3d_ammonium_collapse_surface(None)
        assert fig_none is not None

        # Boundary: all-parasitic surface (all Z < 0)
        shape = (10, 10)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 10, 10))
        @dataclass
        class MockParasiticBifurcation:
            X_nh4: np.ndarray = field(default_factory=lambda: x_grid)
            Y_sweet11: np.ndarray = field(default_factory=lambda: y_grid)
            Z_nhbi: np.ndarray = field(default_factory=lambda: np.full(shape, -0.05))
            critical_nh4_threshold: float = 0.0

        fig_para = visualizer_3d.plot_3d_ammonium_collapse_surface(MockParasiticBifurcation())
        assert fig_para is not None

    def test_plotly_seasonal_trajectory_boundaries(self, sample_exp1_result):
        """Verify 3D seasonal flux space trajectory generation and fallback modes."""
        # Normal
        fig_traj = visualizer_3d.plot_3d_flux_space_trajectory(sample_exp1_result)
        assert fig_traj is not None
        assert len(fig_traj.data) > 0

        # Boundary: None input
        fig_none = visualizer_3d.plot_3d_flux_space_trajectory(None)
        assert fig_none is not None

        # Boundary: custom regimes dict
        custom_exp1 = {
            "Regime Alpha": {
                "suc_syn_r": 10.0,
                "sweet11_export": 2.0,
                "nitrogenase_ab": 1.0,
                "biomass_rice": 0.08,
                "ex_photon_r": 100.0,
                "is_starved": False,
            },
            "Regime Beta": {
                "suc_syn_r": 4.0,
                "sweet11_export": 0.5,
                "nitrogenase_ab": 0.2,
                "biomass_rice": 0.03,
                "ex_photon_r": 30.0,
                "is_starved": True,
            },
        }
        fig_custom = visualizer_3d.plot_3d_flux_space_trajectory(custom_exp1)
        assert fig_custom is not None


# ==============================================================================
# SECTION 3: Performance & Concurrency Stress Testing
# ==============================================================================


class TestPerformanceAndConcurrencyStress:
    """High-throughput execution latency and multi-threaded stress benchmarks."""

    def test_exporter_csv_throughput_and_latency(self):
        """Benchmark generate_fluxes_csv over 1,000 iterations."""
        flux_sample = {f"RXN_{i:02d}": float(i) * 1.25 for i in range(25)}
        n_iters = 1000
        latencies = []

        t_start = time.perf_counter()
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = app.generate_fluxes_csv(flux_sample)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        total_time = time.perf_counter() - t_start

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        ops_per_sec = n_iters / total_time

        print(f"\n[BENCHMARK] generate_fluxes_csv: Mean={mean_lat:.3f}ms, P95={p95_lat:.3f}ms, P99={p99_lat:.3f}ms, Throughput={ops_per_sec:.1f} ops/s")
        assert mean_lat < 5.0, f"Mean latency {mean_lat:.2f}ms exceeded 5.0ms threshold"
        assert ops_per_sec > 200, f"Throughput {ops_per_sec:.1f} ops/s fell below 200 ops/s"

    def test_tensor_csv_throughput_and_latency(self, sample_phase_plane):
        """Benchmark generate_tensor_csv over 100 iterations."""
        n_iters = 100
        latencies = []

        t_start = time.perf_counter()
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = app.generate_tensor_csv(sample_phase_plane)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        total_time = time.perf_counter() - t_start

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        ops_per_sec = n_iters / total_time

        print(f"\n[BENCHMARK] generate_tensor_csv: Mean={mean_lat:.3f}ms, P95={p95_lat:.3f}ms, Throughput={ops_per_sec:.1f} ops/s")
        assert mean_lat < 50.0, f"Mean latency {mean_lat:.2f}ms exceeded 50.0ms threshold"

    def test_threejs_renderer_throughput(self):
        """Benchmark render_threejs_rhizosphere over 1,000 iterations."""
        n_iters = 1000
        latencies = []

        t_start = time.perf_counter()
        for i in range(n_iters):
            t0 = time.perf_counter()
            _ = visualizer_3d.render_threejs_rhizosphere(
                light_pct=float(i % 100),
                sweet11_flux=2.0,
                nh4_flux=1.0,
                is_starved=(i % 10 == 0),
                is_parasitic=(i % 5 == 0),
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total_time = time.perf_counter() - t_start

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        ops_per_sec = n_iters / total_time

        print(f"\n[BENCHMARK] render_threejs_rhizosphere: Mean={mean_lat:.3f}ms, P95={p95_lat:.3f}ms, Throughput={ops_per_sec:.1f} ops/s")
        assert mean_lat < 2.0, f"Three.js HTML render latency {mean_lat:.2f}ms exceeded 2.0ms limit"
        assert ops_per_sec > 500, f"Three.js throughput {ops_per_sec:.1f} ops/s fell below 500 ops/s"

    def test_plotly_scene_generation_latency(self):
        """Benchmark plot_3d_anatomical_scene over 30 iterations."""
        sample_fluxes = {
            "EX_photon_r": 100.0,
            "SWEET11_export": 2.0,
            "NITROGENASE_ab": 1.2,
            "NH4_excretion_ab": 2.4,
            "BIOMASS_rice": 0.085,
            "ATPM_r": 2.50,
        }
        n_iters = 30
        latencies = []

        t_start = time.perf_counter()
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = visualizer_3d.plot_3d_anatomical_scene(sample_fluxes)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total_time = time.perf_counter() - t_start

        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        ops_per_sec = n_iters / total_time

        print(f"\n[BENCHMARK] plot_3d_anatomical_scene: Mean={mean_lat:.3f}ms, P95={p95_lat:.3f}ms, Throughput={ops_per_sec:.1f} ops/s")
        # Interactive standard: 3D Plotly scene generation should complete within 500ms per frame
        assert mean_lat < 500.0, f"Plotly scene build latency {mean_lat:.2f}ms exceeded 500ms interactive threshold"

    def test_concurrent_multi_threaded_export(self, sample_phase_plane):
        """Verify thread safety and concurrency across 8 worker threads."""
        flux_sample = {f"RXN_{i:02d}": float(i) * 1.5 for i in range(25)}

        def worker_task(thread_id: int) -> Tuple[int, int, int]:
            csv1 = app.generate_fluxes_csv(flux_sample)
            csv2 = app.generate_tensor_csv(sample_phase_plane)
            html = visualizer_3d.render_threejs_rhizosphere(70.0, 2.5, 1.2, False, False)
            return len(csv1), len(csv2), len(html)

        n_workers = 8
        n_tasks = 40
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(n_tasks)]
            results = [f.result(timeout=10.0) for f in concurrent.futures.as_completed(futures)]

        assert len(results) == n_tasks
        # All thread executions must return identical lengths (zero race condition corruption)
        first_res = results[0]
        for res in results:
            assert res == first_res, f"Concurrent execution produced non-deterministic output: {res} vs {first_res}"
