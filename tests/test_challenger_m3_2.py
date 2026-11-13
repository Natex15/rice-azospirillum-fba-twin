"""tests/test_challenger_m3_2.py.

Empirical Adversarial Challenge Suite for Milestone M3 (3D Visualization Suite).
Author: Challenger 2 (critic, specialist)

Executes 4 comprehensive stress-testing scenarios:
1. Geometric Mesh Topology & Manifold Checks (Mesh3d, Bézier root curves)
2. Vector Field & Cone Math (Cones, non-zero vectors, zero-flux degeneracy, tail anchoring)
3. Surface Grids & Diverging Colormap Alignment (Dimensions, cmin/cmax/cmid=0, neutral plane, tipping line)
4. Serialization & Rendering Performance (JSON/HTML serialization < 2s, < 5MB, Three.js DOM compliance)
"""

from __future__ import annotations

import html.parser
import math
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

import visualizer_3d
from model_builder import build_two_compartment_model
from simulation_engine import (
    FBASolution,
    PhasePlane2DResult,
    AmmoniumBifurcationResult,
    Experiment1Result,
    simulate_experiment_1_seasonal,
    generate_2d_phase_plane,
    pinpoint_experiment_3_bifurcation,
)

try:
  import plotly.graph_objects as go
except ImportError:
  go = None


@pytest.fixture(scope="module")
def shared_model():
  return build_two_compartment_model()


@pytest.fixture(scope="module")
def standard_fluxes():
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


@pytest.fixture(scope="module")
def phase_plane_data(shared_model):
  return generate_2d_phase_plane(
      shared_model,
      param_x=("light_pct", 20.0, 100.0, 6),
      param_y=("sweet11_export", 0.0, 8.0, 5),
  )


@pytest.fixture(scope="module")
def bifurcation_data(shared_model):
  return pinpoint_experiment_3_bifurcation(
      shared_model, light_pct=50.0, sweet11_export=2.0
  )


@pytest.fixture(scope="module")
def seasonal_data(shared_model):
  return simulate_experiment_1_seasonal(shared_model)


# ==============================================================================
# Scenario 1: Geometric Mesh Topology & Manifold Checks
# ==============================================================================


class TestScenario1MeshTopologyAndBézier:
  """Stress-tests go.Mesh3d topology, face manifold bounds, and Bézier root continuity."""

  @pytest.mark.parametrize("cutaway", [True, False])
  def test_mesh3d_vertex_and_face_topology(self, standard_fluxes, cutaway):
    """Verify that Mesh3d vertices (x,y,z) and face indices (i,j,k) are valid, non-degenerate, and bounded."""
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=standard_fluxes,
        is_starved=False,
        is_parasitic=False,
        show_soil_cutaway=cutaway,
    )
    assert fig is not None

    mesh_traces = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    assert len(mesh_traces) == 1, "Must contain exactly one Mesh3d soil trace"
    mesh = mesh_traces[0]

    x, y, z = np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.z)
    i_idx, j_idx, k_idx = np.asarray(mesh.i), np.asarray(mesh.j), np.asarray(mesh.k)

    # 1. Equal lengths of coordinate arrays
    n_v = len(x)
    assert len(y) == n_v and len(z) == n_v, "Vertex coordinate arrays x, y, z must have equal length"
    assert n_v > 0, "Vertex array must not be empty"

    # 2. Equal lengths of face index arrays
    n_f = len(i_idx)
    assert len(j_idx) == n_f and len(k_idx) == n_f, "Face index arrays i, j, k must have equal length"
    assert n_f > 0, "Face array must not be empty"

    # 3. Valid index bounds [0, n_v)
    assert np.all(i_idx >= 0) and np.all(i_idx < n_v), "i index out of bounds"
    assert np.all(j_idx >= 0) and np.all(j_idx < n_v), "j index out of bounds"
    assert np.all(k_idx >= 0) and np.all(k_idx < n_v), "k index out of bounds"

    # 4. No NaN or Inf coordinates
    assert not np.isnan(x).any() and not np.isinf(x).any(), "NaN/Inf in vertex x"
    assert not np.isnan(y).any() and not np.isinf(y).any(), "NaN/Inf in vertex y"
    assert not np.isnan(z).any() and not np.isinf(z).any(), "NaN/Inf in vertex z"

    # 5. Non-degenerate triangular faces (no vertices duplicated within a single triangle)
    degen_triangles = (i_idx == j_idx) | (j_idx == k_idx) | (i_idx == k_idx)
    assert not degen_triangles.any(), f"Found {np.sum(degen_triangles)} degenerate triangles with duplicated vertices"

    # 6. Non-zero area verification for each triangular face
    v0 = np.column_stack([x[i_idx], y[i_idx], z[i_idx]])
    v1 = np.column_stack([x[j_idx], y[j_idx], z[j_idx]])
    v2 = np.column_stack([x[k_idx], y[k_idx], z[k_idx]])
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    assert np.all(face_areas > 1e-7), f"Found {np.sum(face_areas <= 1e-7)} zero-area colinear/degenerate faces"

    # 7. Coordinate geometric domain checks
    # Radial bounds <= r_soil = 3.2
    r_cyl = np.sqrt(x**2 + y**2)
    assert np.all(r_cyl <= 3.2001), f"Vertex radial distance exceeds soil cylinder radius: max={np.max(r_cyl)}"
    # Z depth bounds in [-6.0, 0.0]
    assert np.all(z >= -6.0001) and np.all(z <= 0.0001), f"Z depths out of [-6, 0] bounds: min={np.min(z)}, max={np.max(z)}"

    # 8. Visual styling properties
    assert 0.0 < mesh.opacity <= 1.0, f"Opacity {mesh.opacity} must be in (0, 1]"
    assert mesh.name == "Soil Rhizosphere Volume"

  def test_root_curve_smoothness_and_branch_continuity(self, standard_fluxes):
    """Verify Bézier/parametric root curve generation: smoothness, continuity across branches, no disjoint gaps."""
    fig = visualizer_3d.plot_3d_anatomical_scene(fluxes=standard_fluxes)
    root_traces = [t for t in fig.data if t.name == "Rice Root Network"]
    assert len(root_traces) == 1, "Must contain Rice Root Network trace"
    root_trace = root_traces[0]

    rx, ry, rz = root_trace.x, root_trace.y, root_trace.z
    assert len(rx) == len(ry) == len(rz)

    # Parse disjoint line segments delimited by None
    segments: List[List[Tuple[float, float, float]]] = []
    current_seg: List[Tuple[float, float, float]] = []
    for x_val, y_val, z_val in zip(rx, ry, rz):
      if x_val is None or y_val is None or z_val is None:
        if current_seg:
          segments.append(current_seg)
          current_seg = []
      else:
        current_seg.append((float(x_val), float(y_val), float(z_val)))
    if current_seg:
      segments.append(current_seg)

    # Expected: 1 seminal root + 6 crown roots + (6 * 2 lateral branches) = 19 segments
    assert len(segments) == 19, f"Expected exactly 19 root branch segments, found {len(segments)}"

    # 1. Seminal root smoothness & direction
    sem_seg = segments[0]
    assert len(sem_seg) == 18, f"Expected 18 points in seminal root, found {len(sem_seg)}"
    # Verify seminal root goes downwards from z=0 towards negative z
    assert sem_seg[0][2] == pytest.approx(0.0, abs=1e-3)
    assert sem_seg[-1][2] == pytest.approx(-5.5, abs=1e-3)

    # Verify each segment has no consecutive duplicate/singular points
    for s_idx, seg in enumerate(segments):
      pts = np.array(seg)
      diffs = np.diff(pts, axis=0)
      step_lens = np.linalg.norm(diffs, axis=1)
      assert np.all(step_lens > 1e-4), f"Segment {s_idx} contains singular duplicate points (zero-length step)"
      assert np.all(step_lens < 1.5), f"Segment {s_idx} has discontinuous jump: max step={np.max(step_lens)}"
      # All root z must be <= 0
      assert np.all(pts[:, 2] <= 1e-4), f"Segment {s_idx} penetrates above ground (z > 0): max z={np.max(pts[:, 2])}"

    # 2. Verify lateral branches connect smoothly to their parent crown root
    # Crown roots and lateral roots are organized per crown root i in [0..5]:
    # Segment 0: Seminal root
    # For crown i:
    #   Segment 1 + 3*i: Crown root i (15 points)
    #   Segment 1 + 3*i + 1: Lateral branch at node j_lat=5 (8 points)
    #   Segment 1 + 3*i + 2: Lateral branch at node j_lat=10 (8 points)
    for cr_idx in range(6):
      crown_seg = segments[1 + 3 * cr_idx]
      assert len(crown_seg) == 15, f"Crown root {cr_idx} must have 15 points"

      # Lateral 1 (j_lat = 5)
      lat1_seg = segments[1 + 3 * cr_idx + 1]
      parent_pt1 = crown_seg[5]
      lat1_start = lat1_seg[0]
      dist1 = math.dist(parent_pt1, lat1_start)
      assert dist1 < 1e-6, f"Crown {cr_idx} lateral branch at j_lat=5 disconnected: dist={dist1}"

      # Lateral 2 (j_lat = 10)
      lat2_seg = segments[1 + 3 * cr_idx + 2]
      parent_pt2 = crown_seg[10]
      lat2_start = lat2_seg[0]
      dist2 = math.dist(parent_pt2, lat2_start)
      assert dist2 < 1e-6, f"Crown {cr_idx} lateral branch at j_lat=10 disconnected: dist={dist2}"


# ==============================================================================
# Scenario 2: Vector Field & Cone Math
# ==============================================================================


class TestScenario2VectorFieldAndCones:
  """Stress-tests 3D go.Cone traces, vector components (u,v,w), tail anchoring, and zero-flux resilience."""

  def test_anatomical_scene_cones_standard_fluxes(self, standard_fluxes):
    """Verify non-zero vector lengths, tail anchoring, and valid coordinates for all cones under positive fluxes."""
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=standard_fluxes,
        is_starved=False,
        is_parasitic=False,
    )
    cone_traces = [t for t in fig.data if isinstance(t, go.Cone)]
    # Under standard conditions: photon cones, sweet11 cones, and nh4 cones are present
    assert len(cone_traces) == 3, f"Expected 3 Cone traces, found {len(cone_traces)}"

    cone_names = {t.name: t for t in cone_traces}
    assert "Solar Photon Influx" in cone_names
    assert "SWEET11 Sucrose Flux" in cone_names
    assert "Diazotrophic NH4+ Flux" in cone_names

    for name, cone in cone_names.items():
      x, y, z = np.asarray(cone.x), np.asarray(cone.y), np.asarray(cone.z)
      u, v, w = np.asarray(cone.u), np.asarray(cone.v), np.asarray(cone.w)

      n_pts = len(x)
      assert len(y) == n_pts and len(z) == n_pts, f"{name}: position arrays length mismatch"
      assert len(u) == n_pts and len(v) == n_pts and len(w) == n_pts, f"{name}: vector arrays length mismatch"
      assert n_pts > 0, f"{name}: empty cone array"

      # No NaN or Inf
      for arr, label in [(x, "x"), (y, "y"), (z, "z"), (u, "u"), (v, "v"), (w, "w")]:
        assert not np.isnan(arr).any(), f"{name}: NaN found in {label}"
        assert not np.isinf(arr).any(), f"{name}: Inf found in {label}"

      # Vector lengths must be strictly positive
      v_lens = np.sqrt(u**2 + v**2 + w**2)
      assert np.all(v_lens > 1e-4), f"{name}: degenerate near-zero vector length: min={np.min(v_lens)}"

      # Tail anchoring
      assert cone.anchor == "tail", f"{name}: anchor must be 'tail', got {cone.anchor}"

  def test_zero_fluxes_do_not_produce_degenerate_cones(self):
    """Verify zero fluxes do not produce degenerate zero-length cone rendering artifacts."""
    zero_fluxes = {
        "EX_photon_r": 0.0,
        "SWEET11_export": 0.0,
        "NITROGENASE_ab": 0.0,
        "NH4_excretion_ab": 0.0,
        "BIOMASS_rice": 0.0,
        "ATPM_r": 2.5,
    }
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=zero_fluxes,
        is_starved=False,
        is_parasitic=False,
    )
    # When all active fluxes are 0.0, no cone traces should be added (omitted to avoid zero-length rendering artifacts)
    cone_traces = [t for t in fig.data if isinstance(t, go.Cone)]
    assert len(cone_traces) == 0, f"Expected 0 Cone traces when fluxes are zero, got {len(cone_traces)}"

  def test_parasitism_suppresses_nh4_cones(self, standard_fluxes):
    """Verify that when is_parasitic=True, NH4+ influx cones are completely suppressed."""
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes=standard_fluxes,
        is_starved=False,
        is_parasitic=True,
    )
    cone_names = [t.name for t in fig.data if isinstance(t, go.Cone)]
    assert "Diazotrophic NH4+ Flux" not in cone_names, "Parasitic state must not render NH4+ influx cones"
    assert "Solar Photon Influx" in cone_names
    assert "SWEET11 Sucrose Flux" in cone_names

  def test_flux_space_trajectory_directional_cones(self, seasonal_data):
    """Extract all go.Cone traces from plot_3d_flux_space_trajectory, verify coordinate arrays, unit vectors, and midpoints."""
    fig = visualizer_3d.plot_3d_flux_space_trajectory(seasonal_data)
    assert fig is not None

    cone_traces = [t for t in fig.data if isinstance(t, go.Cone)]
    # With 4 canonical seasons, there are 4 transition steps in closed loop -> up to 4 cones
    assert len(cone_traces) > 0, "Trajectory must render directional progression cones"

    for idx, cone in enumerate(cone_traces):
      x, y, z = np.asarray(cone.x), np.asarray(cone.y), np.asarray(cone.z)
      u, v, w = np.asarray(cone.u), np.asarray(cone.v), np.asarray(cone.w)

      assert len(x) == len(y) == len(z) == len(u) == len(v) == len(w) == 1
      # Vector length should be normalized unit vector (1.0)
      v_len = math.sqrt(float(u[0])**2 + float(v[0])**2 + float(w[0])**2)
      assert v_len == pytest.approx(1.0, abs=1e-4), f"Cone {idx} vector is not normalized: length={v_len}"
      assert not math.isnan(x[0]) and not math.isnan(y[0]) and not math.isnan(z[0])

  def test_flux_space_trajectory_identical_points_resilience(self):
    """Stress test: If two consecutive seasonal regimes have identical coordinates, norm <= 1e-4 must prevent NaN/zero cones."""
    dummy_regimes = {
        "Peak Dry": {"suc_syn_r": 3.0, "sweet11_export": 1.5, "nitrogenase_ab": 1.0, "biomass_rice": 0.1, "ex_photon_r": 100.0, "is_starved": False},
        "Monsoon": {"suc_syn_r": 3.0, "sweet11_export": 1.5, "nitrogenase_ab": 1.0, "biomass_rice": 0.1, "ex_photon_r": 70.0, "is_starved": False},
    }
    fig = visualizer_3d.plot_3d_flux_space_trajectory(dummy_regimes)
    assert fig is not None
    # Transition between identical 3D points must NOT emit NaN cones
    cone_traces = [t for t in fig.data if isinstance(t, go.Cone)]
    for cone in cone_traces:
      u, v, w = float(cone.u[0]), float(cone.v[0]), float(cone.w[0])
      assert not math.isnan(u) and not math.isnan(v) and not math.isnan(w)
      assert math.sqrt(u**2 + v**2 + w**2) > 1e-4


# ==============================================================================
# Scenario 3: Surface Grids & Diverging Colormap Alignment
# ==============================================================================


class TestScenario3SurfaceGridsAndColormaps:
  """Stress-tests surface grid dimensions, colormap alignment (cmid=0.0), and tipping contours."""

  def test_pareto_surface_grid_dimensions_and_domain(self, phase_plane_data):
    """Verify go.Surface grid dimensions match (len(sweet11), len(light)) and ridge lies within domain."""
    fig = visualizer_3d.plot_3d_pareto_surface(phase_plane_data, current_genotype="WT")
    assert fig is not None

    surface_traces = [t for t in fig.data if isinstance(t, go.Surface)]
    assert len(surface_traces) == 1, "Must contain exactly one Surface trace"
    surf = surface_traces[0]

    # Grid dimensions
    x_grid = np.asarray(surf.x)
    y_grid = np.asarray(surf.y)
    z_grid = np.asarray(surf.z)
    sc_grid = np.asarray(surf.surfacecolor)

    assert z_grid.ndim == 2
    ny, nx = z_grid.shape
    assert x_grid.shape == (ny, nx) or (x_grid.ndim == 1 and len(x_grid) == nx)
    assert y_grid.shape == (ny, nx) or (y_grid.ndim == 1 and len(y_grid) == ny)
    assert sc_grid.shape == (ny, nx)

    # Colormap bounds: cmin=0.0, cmax > 0.0
    assert surf.cmin == 0.0
    assert surf.cmax > 0.0
    assert not np.isnan(z_grid).any()

    # Optimal Pareto Ridge trace
    ridge_traces = [t for t in fig.data if t.name == "Optimal Pareto Ridge"]
    if ridge_traces:
      ridge = ridge_traces[0]
      rx, ry, rz = np.asarray(ridge.x), np.asarray(ridge.y), np.asarray(ridge.z)
      x_min, x_max = np.min(x_grid), np.max(x_grid)
      y_min, y_max = np.min(y_grid), np.max(y_grid)
      assert np.all((rx >= x_min - 1e-3) & (rx <= x_max + 1e-3)), "Ridge x coordinates outside surface domain"
      assert np.all((ry >= y_min - 1e-3) & (ry <= y_max + 1e-3)), "Ridge y coordinates outside surface domain"
      assert not np.isnan(rz).any()

  def test_ammonium_collapse_surface_neutral_plane_and_cmid(self, bifurcation_data):
    """Verify cmin, cmax, and cmid=0.0 alignment, ensure neutral Z=0 plane intersects surface properly."""
    fig = visualizer_3d.plot_3d_ammonium_collapse_surface(bifurcation_data)
    assert fig is not None

    surfaces = [t for t in fig.data if isinstance(t, go.Surface)]
    assert len(surfaces) == 2, f"Expected 2 surfaces (NHBI surface + neutral plane), found {len(surfaces)}"

    nhbi_surf = next(s for s in surfaces if s.name == "Ammonium Collapse Surface")
    neutral_plane = next(s for s in surfaces if "Neutral" in s.name)

    # 1. Diverging colormap centering at cmid=0.0
    colorscale = nhbi_surf.colorscale
    if isinstance(colorscale, str):
      assert "rdbu" in colorscale.lower()
    else:
      # Plotly resolves 'RdBu_r' into explicit RGB tuples: blue -> white -> red
      assert isinstance(colorscale, (tuple, list))
      assert len(colorscale) >= 2
    assert nhbi_surf.cmid == 0.0

    # 2. Neutral Z=0 plane properties
    z_zero = np.asarray(neutral_plane.z)
    assert np.all(z_zero == 0.0), "Neutral reference plane must be identically Z=0"
    assert z_zero.shape == np.asarray(nhbi_surf.z).shape, "Neutral plane must match NHBI grid dimensions"
    assert neutral_plane.opacity == pytest.approx(0.35, abs=0.05)

    # 3. Verify Z=0 plane intersects surface properly (NHBI spans both positive and negative values)
    z_nhbi = np.asarray(nhbi_surf.z)
    z_min, z_max = float(np.min(z_nhbi)), float(np.max(z_nhbi))
    assert z_min < 0.0, f"Expected parasitic negative NHBI regime (min={z_min})"
    assert z_max > 0.0, f"Expected mutualistic positive NHBI regime (max={z_max})"
    assert z_min <= 0.0 <= z_max, "Neutral plane Z=0 must intersect the NHBI surface"

    # 4. Tipping point contour extraction math
    tip_traces = [t for t in fig.data if "Tipping Boundary" in str(t.name)]
    assert len(tip_traces) == 1, "Must contain tipping boundary contour trace"
    tip = tip_traces[0]
    bx, by, bz = np.asarray(tip.x), np.asarray(tip.y), np.asarray(tip.z)
    x_nh4 = np.asarray(nhbi_surf.x)
    y_sw = np.asarray(nhbi_surf.y)
    assert np.all((bx >= np.min(x_nh4) - 1e-3) & (bx <= np.max(x_nh4) + 1e-3))
    assert np.all((by >= np.min(y_sw) - 1e-3) & (by <= np.max(y_sw) + 1e-3))
    # Tipping boundary is rendered at z = 0.001 (just above neutral plane)
    assert np.all(bz == pytest.approx(0.001, abs=1e-3))


# ==============================================================================
# Scenario 4: Serialization & Rendering Performance
# ==============================================================================


class StrictHTMLValidator(html.parser.HTMLParser):
  """Validates balanced tags and structure for self-contained Three.js HTML snippets."""

  VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

  def __init__(self):
    super().__init__()
    self.tag_stack: List[str] = []
    self.errors: List[str] = []

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
    tag_lower = tag.lower()
    if tag_lower not in self.VOID_TAGS:
      self.tag_stack.append(tag_lower)

  def handle_endtag(self, tag: str):
    tag_lower = tag.lower()
    if tag_lower in self.VOID_TAGS:
      return
    if not self.tag_stack:
      self.errors.append(f"Unexpected closing tag </{tag_lower}> with empty stack")
      return
    expected = self.tag_stack.pop()
    if expected != tag_lower:
      self.errors.append(f"Mismatched closing tag: expected </{expected}>, got </{tag_lower}>")


class TestScenario4SerializationAndPerformance:
  """Stress-tests figure serialization latency (< 2.0s), size (< 5MB), and Three.js DOM compliance."""

  def test_figure_serialization_latency_and_size(
      self,
      standard_fluxes,
      phase_plane_data,
      bifurcation_data,
      seasonal_data,
  ):
    """Serialize all 4 figures to JSON and HTML. Ensure time < 2.0s per figure and size < 5MB."""
    figures = [
        ("plot_3d_anatomical_scene", visualizer_3d.plot_3d_anatomical_scene(standard_fluxes)),
        ("plot_3d_pareto_surface", visualizer_3d.plot_3d_pareto_surface(phase_plane_data)),
        ("plot_3d_ammonium_collapse_surface", visualizer_3d.plot_3d_ammonium_collapse_surface(bifurcation_data)),
        ("plot_3d_flux_space_trajectory", visualizer_3d.plot_3d_flux_space_trajectory(seasonal_data)),
    ]

    for name, fig in figures:
      assert fig is not None, f"Figure {name} generation returned None"

      # 1. JSON Serialization benchmark
      t0 = time.perf_counter()
      json_str = fig.to_json()
      json_time = time.perf_counter() - t0
      json_size_mb = len(json_str.encode("utf-8")) / (1024 * 1024)

      assert json_time < 2.0, f"{name}: JSON serialization took {json_time:.3f}s (exceeded 2.0s limit)"
      assert json_size_mb < 5.0, f"{name}: JSON size is {json_size_mb:.2f}MB (exceeded 5.0MB limit)"

      # 2. HTML Serialization benchmark (include_plotlyjs=False)
      t0 = time.perf_counter()
      html_str = fig.to_html(include_plotlyjs=False)
      html_time = time.perf_counter() - t0
      html_size_mb = len(html_str.encode("utf-8")) / (1024 * 1024)

      assert html_time < 2.0, f"{name}: HTML serialization took {html_time:.3f}s (exceeded 2.0s limit)"
      assert html_size_mb < 5.0, f"{name}: HTML size is {html_size_mb:.2f}MB (exceeded 5.0MB limit)"

  def test_render_threejs_rhizosphere_dom_compliance_and_speed(self):
    """Ensure render_threejs_rhizosphere generates compliant HTML that loads cleanly into a DOM parser."""
    test_cases = [
        (100.0, 1.842, 1.150, False, False),   # Optimal
        (30.0, 0.450, 0.200, False, False),    # Chlorotic
        (0.0, 0.0, 0.0, True, False),          # Starved
        (100.0, 4.000, 0.0, False, True),      # Parasitic
        (-20.0, -1.0, -0.5, False, False),     # Negative clamped
    ]

    for light, sw, nh4, starved, parasitic in test_cases:
      t0 = time.perf_counter()
      html_output = visualizer_3d.render_threejs_rhizosphere(
          light_pct=light,
          sweet11_flux=sw,
          nh4_flux=nh4,
          is_starved=starved,
          is_parasitic=parasitic,
          height=650,
      )
      render_time = time.perf_counter() - t0

      assert render_time < 0.1, f"Three.js HTML template rendering took {render_time:.4f}s (> 0.1s)"
      size_kb = len(html_output.encode("utf-8")) / 1024
      assert size_kb < 500.0, f"Three.js HTML size {size_kb:.1f}KB is unreasonably large (> 500KB)"

      # DOM parser validation
      parser = StrictHTMLValidator()
      parser.feed(html_output)
      assert len(parser.errors) == 0, f"HTML DOM validation errors: {parser.errors}"
      assert len(parser.tag_stack) == 0, f"Unclosed HTML tags: {parser.tag_stack}"
