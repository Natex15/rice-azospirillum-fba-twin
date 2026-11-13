"""tests/test_adversarial_m3.py.

Empirical Adversarial Stress Harness for visualizer_3d.py (Milestone M3).
Executed by Challenger 1 to stress-test edge cases, degenerate data inputs,
boundary conditions, and JSON/HTML serializability.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

import visualizer_3d
from simulation_engine import PhasePlane2DResult

try:
  import plotly.graph_objects as go
except ImportError:
  go = None


# ==============================================================================
# 1. Three.js Digital Twin Stress Tests (render_threejs_rhizosphere)
# ==============================================================================


class TestThreeJSRhizosphereAdversarial:
  """Adversarial stress testing for render_threejs_rhizosphere."""

  @pytest.mark.parametrize(
      "light_pct",
      [-100.0, -1.0, 0.0, 0.001, 34.9, 35.0, 100.0, 150.0, 1000.0, 1e6],
  )
  def test_extreme_light_percentages(self, light_pct):
    """Stress test light_pct across extreme negative, threshold, and massive values."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=light_pct,
        sweet11_flux=1.84,
        nh4_flux=1.15,
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    assert isinstance(html, str)
    assert len(html) > 5000
    assert "NaN" not in html
    assert "undefined" not in html
    assert "<div id=\"canvas-container\"></div>" in html
    assert "<div id=\"hud\">" in html
    assert "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" in html
    assert "OrbitControls.js" in html

  @pytest.mark.parametrize(
      "sweet11_flux",
      [-50.0, -0.01, 0.0, 0.0001, 0.01, 1.84, 10.0, 1e4, 1e6],
  )
  def test_extreme_sweet11_fluxes(self, sweet11_flux):
    """Stress test SWEET11 export flux across negative, zero, and huge values."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=80.0,
        sweet11_flux=sweet11_flux,
        nh4_flux=1.15,
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    assert isinstance(html, str)
    assert len(html) > 5000
    assert "NaN" not in html
    assert "undefined" not in html
    assert "numSucrose" in html

  @pytest.mark.parametrize(
      "nh4_flux",
      [-20.0, -0.001, 0.0, 0.0001, 0.01, 1.15, 5.0, 1e4, 1e6],
  )
  def test_extreme_nh4_fluxes(self, nh4_flux):
    """Stress test NH4+ fixation flux across negative, zero, and huge values."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=80.0,
        sweet11_flux=1.84,
        nh4_flux=nh4_flux,
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    assert isinstance(html, str)
    assert len(html) > 5000
    assert "NaN" not in html
    assert "undefined" not in html
    assert "numNh4" in html

  @pytest.mark.parametrize(
      "is_starved,is_parasitic,expected_badge,expected_canopy_color",
      [
          (True, True, "LETHAL CARBON STARVATION", "#582d0e"),
          (False, True, "PARASITIC CARBON DRAIN", "#22c55e"),
          (True, False, "LETHAL CARBON STARVATION", "#582d0e"),
          (False, False, "MUTUALISTIC SYMBIOSIS", "#22c55e"),
      ],
  )
  def test_state_permutations(
      self, is_starved, is_parasitic, expected_badge, expected_canopy_color
  ):
    """Verify biological state badge and canopy coloring under all flag permutations."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=100.0,
        sweet11_flux=1.84,
        nh4_flux=1.15,
        is_starved=is_starved,
        is_parasitic=is_parasitic,
        height=650,
    )
    assert expected_badge in html
    assert expected_canopy_color in html

    if is_parasitic:
      assert "#ef4444" in html  # Crimson bacteria
      assert "const nh4Active = !true" in html or "nh4Active = !true" in html
    else:
      assert "#06b6d4" in html  # Cyan bacteria

  @pytest.mark.parametrize("height", [10, 100, 400, 650, 1200, 2000, 5000])
  def test_canvas_height_variations(self, height):
    """Verify dynamic container height injection into CSS, JS, and resize listener."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=100.0,
        sweet11_flux=1.84,
        nh4_flux=1.15,
        is_starved=False,
        is_parasitic=False,
        height=height,
    )
    assert f"height: {height}px;" in html
    assert f"const h = {height};" in html
    assert f"const h_new = {height};" in html

  def test_nan_input_sanitization(self):
    """Verify float('nan') inputs do not cause unhandled crashes or 'NaN' in JS."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=float("nan"),
        sweet11_flux=float("nan"),
        nh4_flux=float("nan"),
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    assert isinstance(html, str)
    assert len(html) > 5000
    assert "NaN" not in html

  def test_infinite_input_anomaly_detection(self):
    """Detect injection of non-standard 'inf' token into JavaScript code when float('inf') is passed."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=float("inf"),
        sweet11_flux=float("inf"),
        nh4_flux=float("inf"),
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    # Finding: Python f-strings inject raw 'inf' which causes JS ReferenceError: inf is not defined
    raw_inf_lines = [
        line.strip()
        for line in html.splitlines()
        if "inf" in line.lower()
        and "http" not in line
        and "telemetry" not in line
    ]
    assert len(raw_inf_lines) > 0, "Expected to detect unhandled raw 'inf' tokens in JS"


# ==============================================================================
# 2. Plotly 3D Anatomical Scene Stress Tests (plot_3d_anatomical_scene)
# ==============================================================================


class TestPlotly3DAnatomicalSceneAdversarial:
  """Adversarial stress testing for plot_3d_anatomical_scene."""

  def test_none_and_empty_dict_fluxes(self):
    """Verify handling when fluxes is None, empty dict, or missing standard keys."""
    fig_none = visualizer_3d.plot_3d_anatomical_scene(fluxes=None)
    fig_empty = visualizer_3d.plot_3d_anatomical_scene(fluxes={})
    fig_random = visualizer_3d.plot_3d_anatomical_scene(
        fluxes={"unrelated_key": 999.0}
    )

    for fig in [fig_none, fig_empty, fig_random]:
      assert isinstance(fig, go.Figure)
      d = fig.to_dict()
      assert "data" in d and len(d["data"]) >= 5
      j = fig.to_json()
      assert len(j) > 1000

  def test_negative_flux_values(self):
    """Verify negative fluxes are clamped to 0.0 and do not invert geometry."""
    negative_fluxes = {
        "EX_photon_r": -100.0,
        "PHOTOSYN_r": -50.0,
        "SUC_SYN_r": -10.0,
        "SWEET11_export": -5.0,
        "NITROGENASE_ab": -2.0,
        "NH4_excretion_ab": -4.0,
        "BIOMASS_rice": -0.5,
        "ATPM_r": -2.5,
    }
    fig = visualizer_3d.plot_3d_anatomical_scene(fluxes=negative_fluxes)
    assert isinstance(fig, go.Figure)
    d = fig.to_dict()
    assert isinstance(d, dict)
    # Cones should be suppressed when fluxes <= 0
    cone_traces = [t for t in fig.data if t.type == "cone"]
    assert len(cone_traces) == 0

  def test_huge_and_infinite_flux_values(self):
    """Verify extremely high and infinite flux values are handled smoothly."""
    huge_fluxes = {
        "EX_photon_r": 1e6,
        "SWEET11_export": 1e6,
        "NITROGENASE_ab": 1e6,
        "NH4_excretion_ab": 1e6,
        "BIOMASS_rice": 1e6,
        "ATPM_r": 1e6,
    }
    fig_huge = visualizer_3d.plot_3d_anatomical_scene(fluxes=huge_fluxes)
    assert isinstance(fig_huge, go.Figure)
    d_huge = fig_huge.to_dict()
    assert len(d_huge["data"]) >= 5

    # With float('inf')
    inf_fluxes = {
        "EX_photon_r": float("inf"),
        "SWEET11_export": float("inf"),
        "NITROGENASE_ab": float("inf"),
        "NH4_excretion_ab": float("inf"),
        "BIOMASS_rice": float("inf"),
        "ATPM_r": float("inf"),
    }
    fig_inf = visualizer_3d.plot_3d_anatomical_scene(fluxes=inf_fluxes)
    assert isinstance(fig_inf, go.Figure)
    j_inf = fig_inf.to_json()
    assert len(j_inf) > 1000

  def test_nan_flux_values(self):
    """Verify NaN flux values do not crash and produce valid figure."""
    nan_fluxes = {
        "EX_photon_r": float("nan"),
        "SWEET11_export": float("nan"),
        "NITROGENASE_ab": float("nan"),
        "NH4_excretion_ab": float("nan"),
        "BIOMASS_rice": float("nan"),
        "ATPM_r": float("nan"),
    }
    fig_nan = visualizer_3d.plot_3d_anatomical_scene(fluxes=nan_fluxes)
    assert isinstance(fig_nan, go.Figure)
    d_nan = fig_nan.to_dict()
    assert len(d_nan["data"]) >= 3

  def test_soil_cutaway_variations(self):
    """Verify soil cutaway Mesh3d vertices and indices for both True and False."""
    fig_cut = visualizer_3d.plot_3d_anatomical_scene(show_soil_cutaway=True)
    fig_full = visualizer_3d.plot_3d_anatomical_scene(show_soil_cutaway=False)

    mesh_cut = [t for t in fig_cut.data if t.type == "mesh3d"][0]
    mesh_full = [t for t in fig_full.data if t.type == "mesh3d"][0]

    # Verify both have non-empty i, j, k triangle index arrays
    assert len(mesh_cut.i) > 0
    assert len(mesh_full.i) > 0
    assert len(mesh_cut.i) == len(mesh_cut.j) == len(mesh_cut.k)
    assert len(mesh_full.i) == len(mesh_full.j) == len(mesh_full.k)
    # The cutaway geometry includes extra radial wall quads
    assert len(mesh_cut.i) != len(mesh_full.i)


# ==============================================================================
# 3. Plotly 3D Mathematical Surfaces Stress Tests
# ==============================================================================


class TestPlotly3DMathSurfacesAdversarial:
  """Adversarial stress testing for Engine B 3D surfaces."""

  # ----------------------------------------------------------------------------
  # plot_3d_pareto_surface
  # ----------------------------------------------------------------------------

  def test_pareto_surface_all_zeros(self):
    """Verify plot_3d_pareto_surface handles all-zero grids."""
    zero_grid = {
        "X": np.zeros((4, 4)),
        "Y": np.zeros((4, 4)),
        "Z_biomass_rice": np.zeros((4, 4)),
        "Z_nitrogenase_ab": np.zeros((4, 4)),
    }
    fig = visualizer_3d.plot_3d_pareto_surface(zero_grid)
    assert isinstance(fig, go.Figure)
    surface_trace = [t for t in fig.data if t.type == "surface"][0]
    assert surface_trace.cmax == 1e-4  # Clamped minimum to avoid 0 cmax division

  def test_pareto_surface_tiny_2x2_dataclass(self):
    """Verify plot_3d_pareto_surface handles minimal 2x2 grid when passed as PhasePlane2DResult."""
    pp = PhasePlane2DResult(
        param_x_name="Solar Irradiance",
        param_y_name="SWEET11 Exudation",
        x_values=np.array([10.0, 50.0]),
        y_values=np.array([0.0, 4.0]),
        X=np.array([[10.0, 50.0], [10.0, 50.0]]),
        Y=np.array([[0.0, 0.0], [4.0, 4.0]]),
        Z_biomass_rice=np.array([[0.01, 0.05], [0.02, 0.08]]),
        Z_nitrogenase_ab=np.array([[0.1, 0.5], [0.2, 0.9]]),
        Z_nhbi=np.zeros((2, 2)),
        Z_suc_syn_r=np.zeros((2, 2)),
        Z_atpm_r=np.zeros((2, 2)),
        starvation_mask=np.zeros((2, 2), dtype=bool),
        pareto_ridge_x=np.array([10.0, 50.0]),
        pareto_ridge_y=np.array([0.0, 4.0]),
        cliff_x=np.array([10.0, 50.0]),
        cliff_y=np.array([4.0, 4.0]),
    )
    fig = visualizer_3d.plot_3d_pareto_surface(
        pp, current_genotype="OsSWEET11_WT"
    )
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert any("Optimal Pareto Ridge" in n for n in trace_names)
    assert any("Starvation Cliff" in n for n in trace_names)
    assert any("Genotype: WT [ACTIVE]" in n for n in trace_names)

  def test_pareto_surface_dict_polymorphism_anomaly_detection(self):
    """Detect that plot_3d_pareto_surface fails to extract pareto_ridge_x and cliff_x from dicts."""
    tiny_dict = {
        "X": np.array([[10.0, 50.0], [10.0, 50.0]]),
        "Y": np.array([[0.0, 0.0], [4.0, 4.0]]),
        "Z_biomass_rice": np.array([[0.01, 0.05], [0.02, 0.08]]),
        "Z_nitrogenase_ab": np.array([[0.1, 0.5], [0.2, 0.9]]),
        "pareto_ridge_x": [10.0, 50.0],
        "pareto_ridge_y": [0.0, 4.0],
        "cliff_x": [10.0, 50.0],
        "cliff_y": [4.0, 4.0],
    }
    fig = visualizer_3d.plot_3d_pareto_surface(
        tiny_dict, current_genotype="OsSWEET11_WT"
    )
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    # Finding: getattr(dict, 'pareto_ridge_x') returns None, so dict ridges/cliffs are silently ignored!
    has_ridge = any("Optimal Pareto Ridge" in n for n in trace_names)
    assert not has_ridge, "Empirically confirmed: dict-based pareto_ridge_x is not extracted by getattr"

  def test_pareto_surface_with_nans(self):
    """Verify plot_3d_pareto_surface cleans NaNs using nan_to_num."""
    nan_grid = {
        "X": np.array([[10.0, 50.0], [10.0, 50.0]]),
        "Y": np.array([[0.0, 0.0], [4.0, 4.0]]),
        "Z_biomass_rice": np.array([[np.nan, 0.05], [0.02, np.nan]]),
        "Z_nitrogenase_ab": np.array([[np.nan, np.nan], [0.2, 0.9]]),
    }
    fig = visualizer_3d.plot_3d_pareto_surface(nan_grid)
    assert isinstance(fig, go.Figure)
    d = fig.to_dict()
    assert isinstance(d, dict)

  def test_pareto_surface_unknown_genotype(self):
    """Verify unknown or bizarre genotype strings do not cause errors."""
    mock_data = {
        "X": np.linspace(10, 100, 5),
        "Y": np.linspace(0, 8, 5),
        "Z_biomass_rice": np.ones((5, 5)) * 0.1,
        "Z_nitrogenase_ab": np.ones((5, 5)) * 0.5,
    }
    fig = visualizer_3d.plot_3d_pareto_surface(
        mock_data, current_genotype="NON_EXISTENT_MUTANT_XYZ_999"
    )
    assert isinstance(fig, go.Figure)
    # None of the markers should be marked [ACTIVE]
    trace_names = [t.name for t in fig.data]
    assert not any("[ACTIVE]" in n for n in trace_names)

  def test_pareto_surface_none_input(self):
    """Verify None input produces a clean empty dark figure."""
    fig = visualizer_3d.plot_3d_pareto_surface(None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert "No Phase Plane Data Available" in fig.layout.annotations[0].text

  # ----------------------------------------------------------------------------
  # plot_3d_ammonium_collapse_surface
  # ----------------------------------------------------------------------------

  def test_ammonium_collapse_all_positive_benefit(self):
    """Verify bifurcation surface when all NHBI values are positive (no zero crossing)."""
    X, Y = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 8, 5))
    data = {
        "X_nh4": X,
        "Y_sweet11": Y,
        "Z_nhbi": np.full((5, 5), 0.25),
        "critical_nh4_threshold": 5.0,
    }
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3

  def test_ammonium_collapse_all_negative_benefit(self):
    """Verify bifurcation surface when all NHBI values are negative (pure parasitism)."""
    X, Y = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 8, 5))
    data = {
        "X_nh4": X,
        "Y_sweet11": Y,
        "Z_nhbi": np.full((5, 5), -0.15),
        "critical_nh4_threshold": 0.0,
    }
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3

  def test_ammonium_collapse_empty_contour_dataframe(self):
    """Verify bifurcation surface when bifurcation_contour is empty DataFrame."""
    X, Y = np.meshgrid(np.linspace(0, 10, 4), np.linspace(0, 8, 4))
    data = {
        "X_nh4": X,
        "Y_sweet11": Y,
        "Z_nhbi": np.zeros((4, 4)),
        "critical_nh4_threshold": 3.75,
        "bifurcation_contour": pd.DataFrame(),  # Empty DataFrame
    }
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(data)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    # Fallback straight vertical tipping line is drawn
    assert any("Tipping Boundary" in n for n in trace_names)

  def test_ammonium_collapse_none_input(self):
    """Verify None input produces a clean empty dark figure."""
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert "No Bifurcation Data Available" in fig.layout.annotations[0].text

  # ----------------------------------------------------------------------------
  # plot_3d_flux_space_trajectory
  # ----------------------------------------------------------------------------

  def test_flux_space_trajectory_single_season(self):
    """Verify trajectory handles a dictionary with only 1 season without crashing."""
    data = {
        "seasonal_regimes": {
            "Monsoon": {
                "suc_syn_r": 2.5,
                "sweet11_export": 1.2,
                "nitrogenase_ab": 0.8,
                "biomass_rice": 0.09,
                "ex_photon_r": 70.0,
                "is_starved": False,
            }
        }
    }
    fig = visualizer_3d.plot_3d_flux_space_trajectory(data)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert any("Monsoon" in n for n in trace_names)

  def test_flux_space_trajectory_custom_unmatched_names(self):
    """Verify trajectory falls back gracefully when season names do not match canonical set."""
    data = {
        "seasonal_regimes": {
            "Custom Regime Alpha": {
                "suc_syn_r": 1.0,
                "sweet11_export": 0.5,
                "nitrogenase_ab": 0.2,
                "biomass_rice": 0.05,
                "ex_photon_r": 40.0,
                "is_starved": False,
            },
            "Custom Regime Beta": {
                "suc_syn_r": 3.0,
                "sweet11_export": 2.0,
                "nitrogenase_ab": 1.5,
                "biomass_rice": 0.12,
                "ex_photon_r": 90.0,
                "is_starved": False,
            },
        }
    }
    fig = visualizer_3d.plot_3d_flux_space_trajectory(data)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert any("Custom Regime Alpha" in n for n in trace_names)
    assert any("Custom Regime Beta" in n for n in trace_names)

  def test_flux_space_trajectory_all_zero_fluxes(self):
    """Verify trajectory handles all zero fluxes across all regimes without crashing."""
    data = {
        "seasonal_regimes": {
            "Peak Dry": {
                "suc_syn_r": 0.0,
                "sweet11_export": 0.0,
                "nitrogenase_ab": 0.0,
                "biomass_rice": 0.0,
                "ex_photon_r": 0.0,
                "is_starved": True,
            },
            "Monsoon": {
                "suc_syn_r": 0.0,
                "sweet11_export": 0.0,
                "nitrogenase_ab": 0.0,
                "biomass_rice": 0.0,
                "ex_photon_r": 0.0,
                "is_starved": True,
            },
        }
    }
    fig = visualizer_3d.plot_3d_flux_space_trajectory(data)
    assert isinstance(fig, go.Figure)
    d = fig.to_dict()
    assert isinstance(d, dict)

  def test_flux_space_trajectory_empty_regimes_dict(self):
    """Verify trajectory handles an empty seasonal_regimes dict gracefully."""
    data = {"seasonal_regimes": {}}
    fig = visualizer_3d.plot_3d_flux_space_trajectory(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1

  def test_flux_space_trajectory_none_input(self):
    """Verify trajectory handles None input gracefully."""
    fig = visualizer_3d.plot_3d_flux_space_trajectory(None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert "No Seasonal Trajectory Data Available" in fig.layout.annotations[0].text
