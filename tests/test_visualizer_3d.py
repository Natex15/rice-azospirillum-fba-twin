"""tests/test_visualizer_3d.py.

Comprehensive test suite for visualizer_3d.py (Milestone M3: 3D Visualization Suite).
Covers Engine A (Three.js WebGL and Plotly 3D Vector Anatomical Scene) and
Engine B (3D Pareto Surface, 3D Ammonium Collapse Surface, 3D Flux Space Trajectory).
"""

import numpy as np
import pandas as pd
import pytest
import visualizer_3d

try:
  import plotly.graph_objects as go
except ImportError:
  go = None

from simulation_engine import (
    FBASolution,
    PhasePlane2DResult,
    AmmoniumBifurcationResult,
    Experiment1Result,
    Sweep1DResult,
    simulate_experiment_1_seasonal,
    generate_2d_phase_plane,
    pinpoint_experiment_3_bifurcation,
)
from model_builder import build_two_compartment_model


@pytest.fixture(scope="module")
def default_model():
  return build_two_compartment_model()


@pytest.fixture(scope="module")
def mock_fluxes():
  return {
      "EX_photon_r": 100.0,
      "PHOTOSYN_r": 50.0,
      "SUC_SYN_r": 3.125,
      "SWEET11_export": 1.842,
      "NITROGENASE_ab": 1.150,
      "NH4_excretion_ab": 2.300,
      "BIOMASS_rice": 0.125,
      "ATPM_r": 2.500,
  }


# ==============================================================================
# Engine A: Three.js WebGL Digital Twin Tests
# ==============================================================================


class TestThreeJSRhizosphere:
  """Unit and invariant tests for render_threejs_rhizosphere."""

  def test_html_structure_and_cdn_embedding(self):
    """Verify HTML is self-contained, valid, and loads Three.js + OrbitControls."""
    html = visualizer_3d.render_threejs_rhizosphere(
        light_pct=100.0,
        sweet11_flux=1.842,
        nh4_flux=1.150,
        is_starved=False,
        is_parasitic=False,
        height=650,
    )
    assert isinstance(html, str)
    assert len(html) > 500
    assert "three" in html.lower()
    assert "OrbitControls" in html
    assert "canvas-container" in html
    assert "hud" in html
    assert "650px" in html

  def test_physiological_color_states(self):
    """Verify physiological color switching for healthy, chlorotic, and necrotic canopy."""
    # Optimal 100% light
    html_healthy = visualizer_3d.render_threejs_rhizosphere(
        100.0, 1.8, 1.15, is_starved=False, is_parasitic=False
    )
    assert "#22c55e" in html_healthy
    assert "MUTUALISTIC SYMBIOSIS" in html_healthy

    # Low light chlorosis (< 35%)
    html_chlorotic = visualizer_3d.render_threejs_rhizosphere(
        30.0, 0.5, 0.2, is_starved=False, is_parasitic=False
    )
    assert "#eab308" in html_chlorotic

    # Starvation collapse
    html_starved = visualizer_3d.render_threejs_rhizosphere(
        0.0, 0.0, 0.0, is_starved=True, is_parasitic=False
    )
    assert "#582d0e" in html_starved
    assert "LETHAL CARBON STARVATION" in html_starved

  def test_parasitic_symbiosis_warning_state(self):
    """Verify bacterial crimson coloring and telemetry alert under parasitism."""
    html_parasitic = visualizer_3d.render_threejs_rhizosphere(
        100.0, 4.0, 0.0, is_starved=False, is_parasitic=True
    )
    assert "#ef4444" in html_parasitic
    assert "PARASITIC CARBON DRAIN" in html_parasitic

  def test_interactive_controls_and_hud_elements(self):
    """Verify interactive camera preset buttons and 270° cutaway toggle exist."""
    html = visualizer_3d.render_threejs_rhizosphere(100.0, 1.8, 1.15, False, False)
    assert "setView('overview')" in html
    assert "setView('canopy')" in html
    assert "setView('roots')" in html
    assert "toggleCutaway()" in html
    assert "resetView()" in html

  def test_dynamic_particle_systems_scaling(self):
    """Verify all 3 particle systems (photons, sucrose, NH4+) are rendered."""
    html = visualizer_3d.render_threejs_rhizosphere(80.0, 2.5, 1.5, False, False)
    assert "numPhotons" in html
    assert "photonGeo" in html
    assert "sucroseGeo" in html
    assert "nh4Geo" in html

  def test_boundary_and_negative_input_sanitization(self):
    """Verify negative and extreme boundary values are safely clamped."""
    html_neg = visualizer_3d.render_threejs_rhizosphere(
        light_pct=-50.0,
        sweet11_flux=-5.0,
        nh4_flux=-2.0,
        is_starved=False,
        is_parasitic=False,
        height=700,
    )
    assert isinstance(html_neg, str)
    assert "700px" in html_neg
    assert "0.0%" in html_neg  # Clamped to 0


# ==============================================================================
# Engine A: Plotly 3D Anatomical Scene Tests
# ==============================================================================


class TestPlotly3DAnatomicalScene:
  """Unit and invariant tests for plot_3d_anatomical_scene."""

  def test_anatomical_scene_complete_generation(self, mock_fluxes):
    """Verify anatomical scene generates valid go.Figure with Mesh3d, Scatter3d, and Cones."""
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=mock_fluxes,
        is_starved=False,
        is_parasitic=False,
        show_soil_cutaway=True,
    )
    if go is not None:
      assert isinstance(fig, go.Figure)
      trace_types = [t.type for t in fig.data]
      assert "mesh3d" in trace_types
      assert "scatter3d" in trace_types
      assert "cone" in trace_types
      assert len(fig.data) >= 5

  def test_soil_cutaway_toggle(self, mock_fluxes):
    """Verify 90-degree cutaway toggle alters Mesh3d triangle indices."""
    fig_cutaway = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=mock_fluxes, show_soil_cutaway=True
    )
    fig_full = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=mock_fluxes, show_soil_cutaway=False
    )
    mesh_cut = [t for t in fig_cutaway.data if t.type == "mesh3d"][0]
    mesh_full = [t for t in fig_full.data if t.type == "mesh3d"][0]
    # Cutaway mesh includes radial interior walls, so triangle count differs
    assert len(mesh_cut.i) != len(mesh_full.i)

  def test_starvation_and_parasitism_states(self, mock_fluxes):
    """Verify physiological colors and cone suppression on starvation/parasitism."""
    # Starved state
    fig_starved = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=mock_fluxes, is_starved=True, is_parasitic=False
    )
    canopy_trace = [t for t in fig_starved.data if t.name == "Rice Shoot Canopy"][0]
    assert canopy_trace.line.color == "#582d0e"

    # Parasitic state (NH4+ cones suppressed, crimson bacteria)
    fig_parasitic = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=mock_fluxes, is_starved=False, is_parasitic=True
    )
    bact_trace = [t for t in fig_parasitic.data if t.name == "A. brasilense Colonies"][0]
    assert bact_trace.marker.color == "#ef4444"
    cone_names = [t.name for t in fig_parasitic.data if t.type == "cone"]
    assert "Diazotrophic NH4+ Flux" not in cone_names

  def test_empty_or_none_fluxes_resilience(self):
    """Verify scene generates smoothly when fluxes is None or empty dict."""
    fig_empty = visualizer_3d.plot_3d_anatomical_scene(fluxes=None)
    if go is not None:
      assert isinstance(fig_empty, go.Figure)
      assert len(fig_empty.data) >= 3


# ==============================================================================
# Engine B: 3D Mathematical Surfaces Tests
# ==============================================================================


class TestPlotly3DMathSurfaces:
  """Unit and invariant tests for Engine B 3D mathematical surfaces."""

  def test_pareto_surface_with_simulation_result(self, default_model):
    """Verify 3D Pareto surface with live PhasePlane2DResult from simulation engine."""
    phase_res = generate_2d_phase_plane(
        default_model,
        param_x=("light_pct", 20.0, 100.0, 5),
        param_y=("sweet11_export", 0.0, 8.0, 4),
    )
    fig = visualizer_3d.plot_3d_pareto_surface(
        phase_res, current_genotype="OsSWEET11_WT"
    )
    if go is not None:
      assert isinstance(fig, go.Figure)
      trace_types = [t.type for t in fig.data]
      assert "surface" in trace_types
      assert "scatter3d" in trace_types
      trace_names = [t.name for t in fig.data]
      assert any("Optimal Pareto Ridge" in name for name in trace_names)
      assert any("Genotype: WT" in name for name in trace_names)

  def test_pareto_surface_genotype_highlighting(self, default_model):
    """Verify active genotype callout expands marker size and badge text."""
    phase_res = generate_2d_phase_plane(
        default_model,
        param_x=("light_pct", 20.0, 100.0, 4),
        param_y=("sweet11_export", 0.0, 8.0, 3),
    )
    fig = visualizer_3d.plot_3d_pareto_surface(
        phase_res, current_genotype="OsSWEET11_OE"
    )
    oe_trace = [t for t in fig.data if "Genotype: OE" in t.name][0]
    assert oe_trace.marker.size == 14
    assert "[ACTIVE]" in oe_trace.name

  def test_ammonium_collapse_surface_with_bifurcation(self, default_model):
    """Verify 3D Ammonium Collapse surface with live BifurcationResult."""
    bifurc = pinpoint_experiment_3_bifurcation(
        default_model, light_pct=50.0, sweet11_export=2.0
    )
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(bifurc)
    if go is not None:
      assert isinstance(fig, go.Figure)
      # Surface + Neutral Z=0 Plane + Bifurcation line + Marker
      assert len(fig.data) >= 3
      trace_names = [t.name for t in fig.data]
      assert any("Neutral Benefit Plane" in name for name in trace_names)
      assert any("Tipping Boundary" in name for name in trace_names)

  def test_flux_space_trajectory_with_seasonal_result(self, default_model):
    """Verify 3D metabolic state trajectory across 4 Philippine seasons."""
    exp1 = simulate_experiment_1_seasonal(default_model, genotype="WT")
    fig = visualizer_3d.plot_3d_flux_space_trajectory(exp1)
    if go is not None:
      assert isinstance(fig, go.Figure)
      trace_names = [t.name for t in fig.data]
      assert any("Seasonal Trajectory Loop" in name for name in trace_names)
      # Check seasonal milestones
      assert any("Peak Dry" in name for name in trace_names)
      assert any("Typhoon" in name for name in trace_names)

  def test_null_and_empty_inputs_resilience(self):
    """Verify all Engine B functions return valid fallback dark figures on None/empty input."""
    fig_p = visualizer_3d.plot_3d_pareto_surface(None)
    fig_a = visualizer_3d.plot_3d_ammonium_collapse_surface(None)
    fig_t = visualizer_3d.plot_3d_flux_space_trajectory(None)
    if go is not None:
      assert isinstance(fig_p, go.Figure)
      assert isinstance(fig_a, go.Figure)
      assert isinstance(fig_t, go.Figure)
