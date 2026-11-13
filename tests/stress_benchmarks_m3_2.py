"""tests/stress_benchmarks_m3_2.py.

Empirical profiling and adversarial stress probe for Milestone M3.
Collects exact timings, payload sizes, mesh metrics, and edge-case behavior.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import html.parser
import json
import math
import time
import numpy as np
import pandas as pd

import visualizer_3d
from model_builder import build_two_compartment_model
from simulation_engine import (
    generate_2d_phase_plane,
    pinpoint_experiment_3_bifurcation,
    simulate_experiment_1_seasonal,
    PhasePlane2DResult,
    AmmoniumBifurcationResult,
)


def run_benchmarks():
  print("=" * 70)
  print("STARTING EMPIRICAL PROFILING & ADVERSARIAL STRESS PROBES (M3)")
  print("=" * 70)

  model = build_two_compartment_model()

  # ---------------------------------------------------------------------------
  # 1. Mesh Topology & Manifold Metrics
  # ---------------------------------------------------------------------------
  print("\n--- 1. Geometric Mesh Topology & Manifold Analysis ---")
  for cutaway in [True, False]:
    fig = visualizer_3d.plot_3d_anatomical_scene(
        fluxes={"EX_photon_r": 100.0, "SWEET11_export": 1.842, "NITROGENASE_ab": 1.15},
        is_starved=False,
        is_parasitic=False,
        show_soil_cutaway=cutaway,
    )
    mesh = [t for t in fig.data if t.type == "mesh3d"][0]
    x, y, z = np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.z)
    i, j, k = np.asarray(mesh.i), np.asarray(mesh.j), np.asarray(mesh.k)

    v0 = np.column_stack([x[i], y[i], z[i]])
    v1 = np.column_stack([x[j], y[j], z[j]])
    v2 = np.column_stack([x[k], y[k], z[k]])
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    print(f"Cutaway={cutaway}:")
    print(f"  Vertices: {len(x)} | Triangular Faces: {len(i)}")
    print(f"  Min Face Area: {np.min(areas):.6f} | Max Face Area: {np.max(areas):.6f} | Total Area: {np.sum(areas):.4f}")
    print(f"  Radial Bound: [{np.min(np.sqrt(x**2 + y**2)):.2f}, {np.max(np.sqrt(x**2 + y**2)):.2f}] (expected <= 3.20)")
    print(f"  Depth Z Bound: [{np.min(z):.2f}, {np.max(z):.2f}] (expected [-6.00, 0.00])")
    print(f"  Degenerate faces (area <= 1e-7): {np.sum(areas <= 1e-7)}")
    print(f"  Index bounds check: min_idx={min(np.min(i), np.min(j), np.min(k))}, max_idx={max(np.max(i), np.max(j), np.max(k))} (len_x={len(x)})")

  # ---------------------------------------------------------------------------
  # 2. Vector Field & Cone Math
  # ---------------------------------------------------------------------------
  print("\n--- 2. Vector Field & Cone Math Analysis ---")
  # Test with zero fluxes
  fig_zero = visualizer_3d.plot_3d_anatomical_scene(
      fluxes={"EX_photon_r": 0.0, "SWEET11_export": 0.0, "NITROGENASE_ab": 0.0, "NH4_excretion_ab": 0.0},
      is_starved=True,
      is_parasitic=True,
  )
  zero_cones = [t for t in fig_zero.data if t.type == "cone"]
  print(f"Zero fluxes + Starved + Parasitic: Cones rendered = {len(zero_cones)} (expected 0 to avoid zero-length cones)")

  # Test with extreme negative & huge fluxes
  fig_extreme = visualizer_3d.plot_3d_anatomical_scene(
      fluxes={"EX_photon_r": -50.0, "SWEET11_export": -10.0, "NITROGENASE_ab": -5.0},
      is_starved=False,
      is_parasitic=False,
  )
  extreme_cones = [t for t in fig_extreme.data if t.type == "cone"]
  print(f"Negative fluxes (clamped): Cones rendered = {len(extreme_cones)} (expected 0)")

  # Positive fluxes vector lengths
  fig_pos = visualizer_3d.plot_3d_anatomical_scene(
      fluxes={"EX_photon_r": 120.0, "SWEET11_export": 3.5, "NITROGENASE_ab": 2.0, "NH4_excretion_ab": 4.0},
      is_starved=False,
      is_parasitic=False,
  )
  for t in fig_pos.data:
    if t.type == "cone":
      u, v, w = np.asarray(t.u), np.asarray(t.v), np.asarray(t.w)
      lens = np.sqrt(u**2 + v**2 + w**2)
      print(f"  Cone trace '{t.name}': count={len(u)}, min_len={np.min(lens):.4f}, max_len={np.max(lens):.4f}, anchor={t.anchor}")

  # ---------------------------------------------------------------------------
  # 3. Surface Grids & Diverging Colormap Alignment
  # ---------------------------------------------------------------------------
  print("\n--- 3. Surface Grids & Colormap Neutral Plane Intersection ---")
  phase_res = generate_2d_phase_plane(model, param_x=("light_pct", 20.0, 100.0, 6), param_y=("sweet11_export", 0.0, 8.0, 5))
  fig_pareto = visualizer_3d.plot_3d_pareto_surface(phase_res, current_genotype="nonexistent_genotype_fallback_test")
  print("Pareto Surface with unknown genotype: rendered successfully without IndexError")

  bif_res = pinpoint_experiment_3_bifurcation(model, light_pct=50.0, sweet11_export=2.0)
  fig_bif = visualizer_3d.plot_3d_ammonium_collapse_surface(bif_res)
  nhbi_surf = [t for t in fig_bif.data if t.name == "Ammonium Collapse Surface"][0]
  z_nhbi = np.asarray(nhbi_surf.z)
  print(f"Ammonium Collapse Surface: Z_nhbi shape={z_nhbi.shape}, min_Z={np.min(z_nhbi):.4f}, max_Z={np.max(z_nhbi):.4f}")
  print(f"  Neutral Z=0 plane intersects: {np.min(z_nhbi) <= 0.0 <= np.max(z_nhbi)} (min <= 0 <= max)")
  print(f"  Colormap cmid: {nhbi_surf.cmid}")

  # ---------------------------------------------------------------------------
  # 4. Serialization Latency & Payload Sizes
  # ---------------------------------------------------------------------------
  print("\n--- 4. Serialization Latency & Payload Sizes ---")
  exp1_res = simulate_experiment_1_seasonal(model, genotype="WT")
  fig_traj = visualizer_3d.plot_3d_flux_space_trajectory(exp1_res)

  bench_targets = [
      ("plot_3d_anatomical_scene", fig_pos),
      ("plot_3d_pareto_surface", fig_pareto),
      ("plot_3d_ammonium_collapse_surface", fig_bif),
      ("plot_3d_flux_space_trajectory", fig_traj),
  ]

  for name, fig in bench_targets:
    t0 = time.perf_counter()
    j_str = fig.to_json()
    t_json = time.perf_counter() - t0
    size_json_kb = len(j_str.encode("utf-8")) / 1024

    t0 = time.perf_counter()
    h_str = fig.to_html(include_plotlyjs=False)
    t_html = time.perf_counter() - t0
    size_html_kb = len(h_str.encode("utf-8")) / 1024

    print(f"Figure: {name}")
    print(f"  to_json(): {t_json*1000:.2f} ms | size: {size_json_kb:.1f} KB")
    print(f"  to_html(): {t_html*1000:.2f} ms | size: {size_html_kb:.1f} KB")
    assert t_json < 2.0 and size_json_kb < 5120.0
    assert t_html < 2.0 and size_html_kb < 5120.0

  # Three.js render benchmark
  t0 = time.perf_counter()
  three_html = visualizer_3d.render_threejs_rhizosphere(100.0, 1.842, 1.15, False, False, 650)
  t_three = time.perf_counter() - t0
  size_three_kb = len(three_html.encode("utf-8")) / 1024
  print(f"\nThree.js render_threejs_rhizosphere:")
  print(f"  Render time: {t_three*1000:.2f} ms | size: {size_three_kb:.1f} KB")
  assert t_three < 0.1 and size_three_kb < 500.0

  print("\n" + "=" * 70)
  print("ALL PROBES & BENCHMARKS COMPLETED SUCCESSFULLY!")
  print("=" * 70)


if __name__ == "__main__":
  run_benchmarks()
