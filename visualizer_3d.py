"""visualizer_3d.py - 3D Biological & Mathematical Visualization Suite.

Milestone M3: 3D Visualization Suite for Rice-Azospirillum FBA Digital Twin.

Contains:
- Engine A: Interactive 3D Biological Rhizosphere Digital Twin
  * render_threejs_rhizosphere: Self-contained WebGL canvas with procedural
    anatomical geometry (tiller culms, arching leaves, root network, SWEET11
    transporter torus nodes, A. brasilense bacterial clusters), dynamic FBA particle
    flows (photons, sucrose, NH4+), physiological color switching, live 270° cutaway
    toggle, and frosted-glass HUD telemetry overlay.
  * plot_3d_anatomical_scene: Interactive Plotly 3D vector scene with parametric
    cutaway soil cylinder (go.Mesh3d), shoot canopy and branching root axes
    (go.Scatter3d), functional biological markers, and 3D flux cones (go.Cone).
- Engine B: 3D Mathematical Phase Plane Surfaces
  * plot_3d_pareto_surface: Multi-objective Pareto trade-off surface (Light x SWEET11
    x Rice Biomass colored by Nitrogenase) with optimal ridge, starvation cliff, and
    genotype benchmark markers.
  * plot_3d_ammonium_collapse_surface: Net Host Benefit (NHBI) bifurcation surface
    with diverging colormap (RdBu_r) centered at Z=0, semi-transparent neutral plane,
    and [NH4+]crit tipping point markers.
  * plot_3d_flux_space_trajectory: 3D metabolic state trajectory across 4 canonical
    Philippine seasonal regimes (Dry, Monsoon, Rainy, Typhoon) with directional cones.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
  import plotly.graph_objects as go
except ImportError:
  go = None


# ==============================================================================
# Helper Utilities
# ==============================================================================


def _create_empty_dark_figure(message: str) -> Any:
  """Returns an empty dark figure with an informative status annotation."""
  if go is None:
    return None
  fig = go.Figure()
  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="#0b0f19",
      plot_bgcolor="#0b0f19",
      scene=dict(
          xaxis=dict(visible=False),
          yaxis=dict(visible=False),
          zaxis=dict(visible=False),
      ),
      annotations=[
          dict(
              text=message,
              xref="paper",
              yref="paper",
              x=0.5,
              y=0.5,
              showarrow=False,
              font=dict(size=14, color="#94a3b8"),
          )
      ],
  )
  return fig


def _normalize_genotype(name: Optional[str]) -> str:
  """Normalizes genotype string for safe matching."""
  if not name:
    return ""
  return (
      str(name)
      .strip()
      .lower()
      .replace("_", "")
      .replace(" ", "")
      .replace("-", "")
  )


# ==============================================================================
# Engine A: Interactive 3D Biological Rhizosphere Digital Twin
# ==============================================================================


def render_threejs_rhizosphere(
    light_pct: float,
    sweet11_flux: float,
    nh4_flux: float,
    is_starved: bool,
    is_parasitic: bool,
    height: int = 650,
) -> str:
  """Renders a self-contained Three.js WebGL 3D Rhizosphere Digital Twin HTML string.

  Designed to be embedded in Streamlit via streamlit.components.v1.html(html_str,
  height=height).

  Parameters
  ----------
  light_pct : float
      Simulated Philippine seasonal solar irradiance percentage [0..150%].
  sweet11_flux : float
      Steady-state SWEET11 phloem-to-rhizosphere export flux (mmol/gDW/h).
  nh4_flux : float
      Diazotrophic biological nitrogen fixation flux (NITROGENASE_ab)
      (mmol/gDW/h).
  is_starved : bool
      Lethal carbon starvation flag (ATPM_r cellular maintenance deficit).
  is_parasitic : bool
      Ecological status flag indicating host carbon parasitism (NHBI < 0).
  height : int, optional
      Height of the rendering canvas container in pixels (default 650).

  Returns
  -------
  str
      Self-contained, production-grade HTML5/JavaScript application string.
  """
  # Robust input sanitization
  light_pct = max(0.0, float(light_pct))
  sweet11_flux = max(0.0, float(sweet11_flux))
  nh4_flux = max(0.0, float(nh4_flux))
  is_starved = bool(is_starved)
  is_parasitic = bool(is_parasitic)
  height = int(height)

  # Physiological canopy pigmentation
  if is_starved:
    canopy_color = "#582d0e"  # Brittle necrotic umber
  elif light_pct < 35.0:
    canopy_color = "#eab308"  # Chlorotic lime-yellow
  else:
    canopy_color = "#22c55e"  # Vibrant photosynthetic emerald

  # Bacterial symbiosis color state
  bacteria_color = "#ef4444" if is_parasitic else "#06b6d4"
  bacteria_emissive = "#b91c1c" if is_parasitic else "#059669"
  bact_emissive_intensity = 1.8 if is_parasitic else 0.8
  is_parasitic_js = "true" if is_parasitic else "false"
  is_starved_js = "true" if is_starved else "false"
  light_flux_val = light_pct * 1.0

  # Ecological status badge
  if is_starved:
    badge_text = "LETHAL CARBON STARVATION"
    badge_class = "badge-starved"
  elif is_parasitic:
    badge_text = "PARASITIC CARBON DRAIN"
    badge_class = "badge-parasitic"
  else:
    badge_text = "MUTUALISTIC SYMBIOSIS"
    badge_class = "badge-mutualist"

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>3D Biological Rhizosphere Digital Twin</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
      background: #070b14; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #f8fafc; user-select: none;
    }}
    #canvas-container {{ position: absolute; top: 0; left: 0; width: 100%; height: {height}px; }}
    #hud {{
      position: absolute; top: 14px; left: 14px; z-index: 10; max-width: 440px;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 14px 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); pointer-events: auto;
    }}
    .hud-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px; }}
    .hud-title {{ font-size: 13px; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 6px; }}
    .hud-badge {{ font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 9999px; letter-spacing: 0.5px; text-transform: uppercase; }}
    .badge-mutualist {{ background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; }}
    .badge-starved {{ background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fbbf24; }}
    .badge-parasitic {{ background: rgba(239, 68, 68, 0.25); border: 1px solid #ef4444; color: #fca5a5; }}
    .hud-telemetry {{ display: grid; grid-template-columns: 1fr; gap: 6px; font-size: 12px; margin-bottom: 12px; }}
    .telemetry-row {{ display: flex; justify-content: space-between; align-items: center; color: #94a3b8; }}
    .telemetry-value {{ font-family: monospace; font-weight: 600; color: #f8fafc; }}
    .hud-controls {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }}
    .hud-btn {{ background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255, 255, 255, 0.15); color: #e2e8f0; border-radius: 6px; padding: 5px 9px; font-size: 11px; font-weight: 500; cursor: pointer; transition: all 0.15s ease; }}
    .hud-btn:hover {{ background: rgba(51, 65, 85, 0.95); border-color: rgba(255, 255, 255, 0.3); color: #ffffff; }}
    .hud-btn-active {{ background: rgba(14, 165, 233, 0.3); border-color: #0ea5e9; color: #38bdf8; }}
    .hud-legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 10px; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; }}
    .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
    .dot-gold {{ background: #fde047; box-shadow: 0 0 6px #fde047; }}
    .dot-cyan {{ background: #38bdf8; box-shadow: 0 0 6px #38bdf8; }}
    .dot-emerald {{ background: #10b981; box-shadow: 0 0 6px #10b981; }}
    .dot-root {{ background: #fef08a; }}
    .dot-bact {{ background: {bacteria_color}; box-shadow: 0 0 6px {bacteria_color}; }}
    #fallback-msg {{ display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(30, 41, 59, 0.95); border: 1px solid #ef4444; border-radius: 8px; padding: 20px; color: #f8fafc; text-align: center; z-index: 100; }}
  </style>
</head>
<body>
  <div id="canvas-container"></div>
  <div id="hud">
    <div class="hud-header">
      <span class="hud-title">🌿 3D Rhizosphere Digital Twin</span>
      <span class="hud-badge {badge_class}">{badge_text}</span>
    </div>
    <div class="hud-telemetry">
      <div class="telemetry-row"><span>☀️ Solar Irradiance:</span><span class="telemetry-value">{light_pct:.1f}% ({light_flux_val:.1f} mmol/gDW/h)</span></div>
      <div class="telemetry-row"><span>🍯 OsSWEET11 Sucrose Flux:</span><span class="telemetry-value">{sweet11_flux:.2f} mmol/gDW/h</span></div>
      <div class="telemetry-row"><span>⚡ Fixed Ammonium (NH₄⁺):</span><span class="telemetry-value">{nh4_flux:.2f} mmol/gDW/h</span></div>
    </div>
    <div class="hud-controls">
      <button class="hud-btn" onclick="setView('overview')">🔄 Overview</button>
      <button class="hud-btn" onclick="setView('canopy')">🌿 Canopy</button>
      <button class="hud-btn" onclick="setView('roots')">🌱 Roots</button>
      <button class="hud-btn hud-btn-active" id="btn-cutaway" onclick="toggleCutaway()">✂️ Cutaway: ON</button>
      <button class="hud-btn" onclick="resetView()">⏪ Reset</button>
    </div>
    <div class="hud-legend">
      <div class="legend-item"><span class="dot dot-gold"></span> Photons</div>
      <div class="legend-item"><span class="dot dot-cyan"></span> Sucrose</div>
      <div class="legend-item"><span class="dot dot-emerald"></span> Fixed NH₄⁺</div>
      <div class="legend-item"><span class="dot dot-root"></span> Roots</div>
      <div class="legend-item"><span class="dot dot-bact"></span> A. brasilense</div>
    </div>
  </div>
  <div id="fallback-msg">WebGL rendering is not supported or failed to initialize in this environment.</div>

  <script>
    if (!window.WebGLRenderingContext) {{
      document.getElementById('fallback-msg').style.display = 'block';
    }} else {{
      const container = document.getElementById('canvas-container');
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x070b14);

      const width = container.clientWidth || window.innerWidth;
      const h = {height};
      const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 1000);
      camera.position.set(14, 4, 18);

      const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: 'high-performance' }});
      renderer.setSize(width, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 4;
      controls.maxDistance = 45;
      controls.target.set(0, -2, 0);
      controls.update();

      const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambLight);

      const sunLight = new THREE.DirectionalLight(0xfffaed, Math.max(0.3, 0.3 + 0.9 * ({light_pct} / 100.0)));
      sunLight.position.set(10, 20, 10);
      sunLight.castShadow = true;
      scene.add(sunLight);

      const rhizoLight = new THREE.PointLight(0x38bdf8, 0.5, 25);
      rhizoLight.position.set(0, -5, 0);
      scene.add(rhizoLight);

      const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x1e1b4b, 0.4);
      scene.add(hemiLight);

      // Overcast cloud deck (opacity scales as light decreases)
      const cloudOpacity = Math.max(0.0, Math.min(0.85, (100.0 - {light_pct}) / 100.0 * 0.75));
      if (cloudOpacity > 0.01) {{
        const cloudGeo = new THREE.RingGeometry(0.1, 10, 36);
        const cloudMat = new THREE.MeshBasicMaterial({{
          color: 0x334155, transparent: true, opacity: cloudOpacity, side: THREE.DoubleSide
        }});
        const cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
        cloudMesh.rotation.x = -Math.PI / 2;
        cloudMesh.position.set(0, 14, 0);
        scene.add(cloudMesh);
      }}

      // Ground interface disk & collar ring
      const groundGeo = new THREE.CircleGeometry(6.5, 48);
      const groundMat = new THREE.MeshStandardMaterial({{
        color: 0x111827, roughness: 0.9, transparent: true, opacity: 0.7, side: THREE.DoubleSide
      }});
      const groundMesh = new THREE.Mesh(groundGeo, groundMat);
      groundMesh.rotation.x = -Math.PI / 2;
      scene.add(groundMesh);

      const collarGeo = new THREE.RingGeometry(4.75, 4.85, 48);
      const collarMat = new THREE.MeshBasicMaterial({{ color: 0x3b82f6, transparent: true, opacity: 0.4, side: THREE.DoubleSide }});
      const collarMesh = new THREE.Mesh(collarGeo, collarMat);
      collarMesh.rotation.x = -Math.PI / 2;
      collarMesh.position.y = 0.02;
      scene.add(collarMesh);

      // Translucent rhizosphere soil cylinder (interactive 270° cutaway)
      const cutawayGeo = new THREE.CylinderGeometry(4.8, 4.8, 11, 48, 1, false, Math.PI * 0.25, Math.PI * 1.5);
      const fullGeo = new THREE.CylinderGeometry(4.8, 4.8, 11, 48, 1, false, 0, Math.PI * 2.0);
      const soilMat = new THREE.MeshStandardMaterial({{
        color: 0x241812, roughness: 0.85, transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide
      }});
      let isCutaway = true;
      const soilMesh = new THREE.Mesh(cutawayGeo, soilMat);
      soilMesh.position.set(0, -5.5, 0);
      scene.add(soilMesh);

      // Shoot canopy: central tiller culms
      const canopyColor = '{canopy_color}';
      const canopyMat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(canopyColor), roughness: 0.45, metalness: 0.1, side: THREE.DoubleSide
      }});
      const culmConfigs = [
        {{ top: [0.0, 9.2, 0.0], mid: [0.1, 4.5, 0.0] }},
        {{ top: [0.4, 8.5, 0.3], mid: [0.3, 4.0, 0.2] }},
        {{ top: [-0.4, 8.2, 0.2], mid: [-0.3, 3.8, 0.1] }},
        {{ top: [0.2, 7.8, -0.4], mid: [0.15, 3.5, -0.25] }},
        {{ top: [-0.3, 7.5, -0.3], mid: [-0.2, 3.2, -0.15] }}
      ];
      culmConfigs.forEach(c => {{
        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(c.mid[0], c.mid[1], c.mid[2]),
          new THREE.Vector3(c.top[0], c.top[1], c.top[2])
        ]);
        scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 20, 0.08, 8, false), canopyMat));
      }});

      // Arching leaf blades
      const leafConfigs = [
        {{ start: [0.0, 2.0, 0.0], apex: [1.8, 6.5, 0.4], tip: [3.4, 4.2, 0.8] }},
        {{ start: [0.0, 3.0, 0.0], apex: [-1.6, 7.2, 0.6], tip: [-3.2, 4.8, 1.2] }},
        {{ start: [0.0, 3.8, 0.0], apex: [0.5, 7.8, 1.8], tip: [1.0, 5.0, 3.5] }},
        {{ start: [0.0, 2.5, 0.0], apex: [-0.6, 6.8, -1.8], tip: [-1.2, 4.5, -3.4] }},
        {{ start: [0.0, 4.5, 0.0], apex: [2.0, 8.0, -1.2], tip: [3.8, 5.5, -2.2] }},
        {{ start: [0.0, 4.0, 0.0], apex: [-1.8, 7.5, -1.0], tip: [-3.5, 5.0, -1.8] }},
        {{ start: [0.0, 5.2, 0.0], apex: [1.4, 8.5, 1.2], tip: [2.6, 6.2, 2.4] }},
        {{ start: [0.0, 5.5, 0.0], apex: [-1.2, 8.2, 1.0], tip: [-2.4, 5.8, 2.0] }}
      ];
      leafConfigs.forEach(l => {{
        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(l.start[0], l.start[1], l.start[2]),
          new THREE.Vector3(l.apex[0], l.apex[1], l.apex[2]),
          new THREE.Vector3(l.tip[0], l.tip[1], l.tip[2])
        ]);
        scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.05, 6, false), canopyMat));
      }});

      // Panicle seed florets
      const grainMat = new THREE.MeshStandardMaterial({{ color: 0xd97706, roughness: 0.6 }});
      for (let i = 0; i < 16; i++) {{
        const floret = new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 6), grainMat);
        const t = i / 16;
        floret.position.set(0.2 * Math.sin(t * 8) + t * 0.4, 9.2 - t * 1.5, 0.2 * Math.cos(t * 8) + t * 0.3);
        scene.add(floret);
      }}

      // Root architecture: 4 primary crown roots
      const rootMat = new THREE.MeshStandardMaterial({{
        color: 0xfef08a, roughness: 0.5, metalness: 0.05
      }});
      const crownCurves = [
        new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.15, -3.0, 0.1),
          new THREE.Vector3(0.2, -7.0, -0.15), new THREE.Vector3(0.1, -10.5, 0.0)
        ]),
        new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0), new THREE.Vector3(-0.25, -2.8, 0.2),
          new THREE.Vector3(-0.5, -6.5, 0.3), new THREE.Vector3(-0.35, -9.8, 0.2)
        ]),
        new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.1, -2.5, -0.25),
          new THREE.Vector3(-0.2, -6.2, -0.5), new THREE.Vector3(-0.1, -10.0, -0.4)
        ]),
        new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0), new THREE.Vector3(-0.15, -2.7, -0.2),
          new THREE.Vector3(0.35, -6.8, 0.25), new THREE.Vector3(0.25, -10.2, 0.1)
        ])
      ];
      crownCurves.forEach(c => {{
        scene.add(new THREE.Mesh(new THREE.TubeGeometry(c, 24, 0.11, 8, false), rootMat));
      }});

      // 16 lateral root branches & elongation zone positions
      const lateralPaths = [];
      const sweet11Nodes = [];
      const rootTipPositions = [];
      const numLaterals = 16;
      for (let i = 0; i < numLaterals; i++) {{
        const angle = i * (Math.PI * 2 / numLaterals) + (i % 2) * 0.2;
        const depth = -1.2 - (i / numLaterals) * 7.5;
        const rCrown = 0.2 + (depth / -10.5) * 0.25;
        const p0 = new THREE.Vector3(rCrown * Math.cos(angle), depth, rCrown * Math.sin(angle));
        const midDist = 1.6 + (i % 3) * 0.5;
        const p1 = new THREE.Vector3(midDist * Math.cos(angle), depth - 0.9, midDist * Math.sin(angle));
        const maxDist = 2.8 + (i % 4) * 0.3;
        const p2 = new THREE.Vector3(maxDist * Math.cos(angle), depth - 1.8, maxDist * Math.sin(angle));

        const fullCurve = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), p0, p1, p2]);
        lateralPaths.push(fullCurve);
        rootTipPositions.push(p2);

        scene.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([p0, p1, p2]), 16, 0.05, 6, false), rootMat));
        sweet11Nodes.push(fullCurve.getPoint(0.8));
      }}

      // OsSWEET11 Transporter rings
      const sweetGlow = {sweet11_flux} > 0.01 ? Math.min(2.0, 0.2 + {sweet11_flux} * 0.45) : 0.05;
      const sweetMat = new THREE.MeshStandardMaterial({{
        color: {sweet11_flux} > 0.01 ? 0x38bdf8 : 0x64748b,
        emissive: {sweet11_flux} > 0.01 ? 0x0284c7 : 0x1e293b,
        emissiveIntensity: sweetGlow,
        roughness: 0.3
      }});
      const torusGeo = new THREE.TorusGeometry(0.12, 0.04, 12, 24);
      sweet11Nodes.forEach((pos, idx) => {{
        const torus = new THREE.Mesh(torusGeo, sweetMat);
        torus.position.copy(pos);
        torus.rotation.x = Math.PI / 2;
        torus.rotation.y = idx * (Math.PI / 8);
        scene.add(torus);
      }});

      // Azospirillum brasilense bacterial colonies (instanced capsules)
      const bactColor = '{bacteria_color}';
      const bactEmissive = '{bacteria_emissive}';
      const bactMat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(bactColor),
        emissive: new THREE.Color(bactEmissive),
        emissiveIntensity: {bact_emissive_intensity},
        roughness: 0.35
      }});
      const numBact = 140;
      const bactMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.08, 8, 8), bactMat, numBact);
      const dummy = new THREE.Object3D();
      const bactPositions = [];
      for (let i = 0; i < numBact; i++) {{
        const tip = rootTipPositions[i % rootTipPositions.length];
        const jitterR = 0.2 + ((i * 37) % 70) * 0.01;
        const jitterTheta = (i * 2.39996);
        const jitterY = (((i * 13) % 80) - 40) * 0.01;
        const bx = tip.x + jitterR * Math.cos(jitterTheta);
        const by = Math.max(-10.5, Math.min(-0.8, tip.y + jitterY));
        const bz = tip.z + jitterR * Math.sin(jitterTheta);
        const bPos = new THREE.Vector3(bx, by, bz);
        bactPositions.push(bPos);

        dummy.position.copy(bPos);
        dummy.scale.set(1.0, 1.4, 1.0);
        dummy.rotation.set((i * 0.7) % Math.PI, (i * 1.3) % Math.PI, 0);
        dummy.updateMatrix();
        bactMesh.setMatrixAt(i, dummy.matrix);
      }}
      bactMesh.instanceMatrix.needsUpdate = true;
      scene.add(bactMesh);

      // Particle System 1: Solar Photons
      const numPhotons = Math.floor(250 * Math.max(0.1, {light_pct} / 100.0));
      const photonGeo = new THREE.BufferGeometry();
      const photonPos = new Float32Array(numPhotons * 3);
      for (let i = 0; i < numPhotons; i++) {{
        const r = ((i * 17) % 50) * 0.1;
        const th = (i * 2.39996);
        photonPos[i * 3] = r * Math.cos(th);
        photonPos[i * 3 + 1] = 0.5 + ((i * 29) % 140) * 0.1;
        photonPos[i * 3 + 2] = r * Math.sin(th);
      }}
      photonGeo.setAttribute('position', new THREE.BufferAttribute(photonPos, 3));
      const photonMat = new THREE.PointsMaterial({{
        color: 0xfde047, size: 0.16, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending
      }});
      const photonPoints = new THREE.Points(photonGeo, photonMat);
      scene.add(photonPoints);

      // Particle System 2: Sucrose Phloem Translocation
      const sucroseActive = {sweet11_flux} > 0.001;
      const numSucrose = sucroseActive ? Math.min(200, Math.floor(30 + {sweet11_flux} * 35)) : 0;
      const sucroseGeo = new THREE.BufferGeometry();
      const sucrosePos = new Float32Array(Math.max(1, numSucrose) * 3);
      const sucroseT = new Float32Array(Math.max(1, numSucrose));
      const sucroseCurveIdx = new Int32Array(Math.max(1, numSucrose));
      for (let i = 0; i < numSucrose; i++) {{
        sucroseT[i] = ((i * 31) % 100) * 0.01;
        sucroseCurveIdx[i] = i % lateralPaths.length;
        const pt = lateralPaths[sucroseCurveIdx[i]].getPoint(sucroseT[i]);
        sucrosePos[i * 3] = pt.x;
        sucrosePos[i * 3 + 1] = pt.y;
        sucrosePos[i * 3 + 2] = pt.z;
      }}
      sucroseGeo.setAttribute('position', new THREE.BufferAttribute(sucrosePos, 3));
      const sucroseMat = new THREE.PointsMaterial({{
        color: 0x38bdf8, size: 0.13, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending
      }});
      const sucrosePoints = new THREE.Points(sucroseGeo, sucroseMat);
      sucrosePoints.visible = sucroseActive;
      scene.add(sucrosePoints);

      // Particle System 3: Fixed NH4+ Influx
      const nh4Active = !{is_parasitic_js} && {nh4_flux} > 0.001;
      const numNh4 = nh4Active ? Math.min(200, Math.floor(30 + {nh4_flux} * 40)) : 0;
      const nh4Geo = new THREE.BufferGeometry();
      const nh4Pos = new Float32Array(Math.max(1, numNh4) * 3);
      const nh4U = new Float32Array(Math.max(1, numNh4));
      const nh4BactIdx = new Int32Array(Math.max(1, numNh4));
      for (let i = 0; i < numNh4; i++) {{
        nh4U[i] = ((i * 23) % 100) * 0.01;
        nh4BactIdx[i] = i % bactPositions.length;
        const bPos = bactPositions[nh4BactIdx[i]];
        const rPos = rootTipPositions[nh4BactIdx[i] % rootTipPositions.length];
        const u = nh4U[i];
        nh4Pos[i * 3] = bPos.x + (rPos.x - bPos.x) * u;
        nh4Pos[i * 3 + 1] = bPos.y + (rPos.y - bPos.y) * u;
        nh4Pos[i * 3 + 2] = bPos.z + (rPos.z - bPos.z) * u;
      }}
      nh4Geo.setAttribute('position', new THREE.BufferAttribute(nh4Pos, 3));
      const nh4Mat = new THREE.PointsMaterial({{
        color: 0x10b981, size: 0.15, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending
      }});
      const nh4Points = new THREE.Points(nh4Geo, nh4Mat);
      nh4Points.visible = nh4Active;
      scene.add(nh4Points);

      // Camera views & cutaway control
      const clock = new THREE.Clock();
      let targetCamPos = null;
      let targetLookAt = null;

      window.setView = function(v) {{
        if (v === 'overview') {{ targetCamPos = new THREE.Vector3(14, 4, 18); targetLookAt = new THREE.Vector3(0, -2, 0); }}
        else if (v === 'canopy') {{ targetCamPos = new THREE.Vector3(0, 10, 14); targetLookAt = new THREE.Vector3(0, 4, 0); }}
        else if (v === 'roots') {{ targetCamPos = new THREE.Vector3(8, -5, 10); targetLookAt = new THREE.Vector3(0, -5, 0); }}
      }};

      window.resetView = function() {{ setView('overview'); }};

      window.toggleCutaway = function() {{
        isCutaway = !isCutaway;
        soilMesh.geometry = isCutaway ? cutawayGeo : fullGeo;
        const btn = document.getElementById('btn-cutaway');
        if (btn) {{
          btn.innerText = isCutaway ? '✂️ Cutaway: ON' : '✂️ Cutaway: OFF';
          if (isCutaway) btn.classList.add('hud-btn-active');
          else btn.classList.remove('hud-btn-active');
        }}
      }};

      function animate() {{
        requestAnimationFrame(animate);
        const delta = clock.getDelta();
        const elapsed = clock.getElapsedTime();

        if (targetCamPos && targetLookAt) {{
          camera.position.lerp(targetCamPos, 0.06);
          controls.target.lerp(targetLookAt, 0.06);
          if (camera.position.distanceTo(targetCamPos) < 0.1) {{
            targetCamPos = null; targetLookAt = null;
          }}
        }}
        controls.update();

        // Pulsing bacterial glow under parasitic warning
        if ({is_parasitic_js}) {{
          const pulse = 1.0 + 0.5 * Math.sin(elapsed * 5.0);
          bactMat.emissiveIntensity = pulse * 1.5;
        }}

        // Update solar photons cascade
        if ({light_pct} > 0.0) {{
          const pArr = photonGeo.attributes.position.array;
          const speed = (0.08 + 0.16 * ({light_pct} / 100.0)) * delta * 60;
          for (let i = 0; i < numPhotons; i++) {{
            pArr[i * 3 + 1] -= speed;
            if (pArr[i * 3 + 1] < 0.2) {{
              pArr[i * 3 + 1] = 14.0 + ((i * 19) % 30) * 0.1;
              const r = ((i * 23) % 50) * 0.1;
              const th = (i * 2.39996);
              pArr[i * 3] = r * Math.cos(th);
              pArr[i * 3 + 2] = r * Math.sin(th);
            }}
          }}
          photonGeo.attributes.position.needsUpdate = true;
        }}

        // Update sucrose phloem particle flow
        if (sucroseActive && numSucrose > 0) {{
          const sArr = sucroseGeo.attributes.position.array;
          const speed = (0.006 + 0.016 * Math.min({sweet11_flux} / 3.0, 2.5)) * delta * 60;
          for (let i = 0; i < numSucrose; i++) {{
            sucroseT[i] += speed;
            if (sucroseT[i] > 1.0) sucroseT[i] = 0.0;
            const pt = lateralPaths[sucroseCurveIdx[i]].getPoint(sucroseT[i]);
            sArr[i * 3] = pt.x;
            sArr[i * 3 + 1] = pt.y;
            sArr[i * 3 + 2] = pt.z;
          }}
          sucroseGeo.attributes.position.needsUpdate = true;
        }}

        // Update fixed NH4+ particle flow
        if (nh4Active && numNh4 > 0) {{
          const nArr = nh4Geo.attributes.position.array;
          const speed = (0.008 + 0.018 * Math.min({nh4_flux} / 2.0, 2.5)) * delta * 60;
          for (let i = 0; i < numNh4; i++) {{
            nh4U[i] += speed;
            if (nh4U[i] > 1.0) nh4U[i] = 0.0;
            const bPos = bactPositions[nh4BactIdx[i]];
            const rPos = rootTipPositions[nh4BactIdx[i] % rootTipPositions.length];
            const u = nh4U[i];
            nArr[i * 3] = bPos.x + (rPos.x - bPos.x) * u;
            nArr[i * 3 + 1] = bPos.y + (rPos.y - bPos.y) * u;
            nArr[i * 3 + 2] = bPos.z + (rPos.z - bPos.z) * u;
          }}
          nh4Geo.attributes.position.needsUpdate = true;
        }}

        renderer.render(scene, camera);
      }}
      animate();

      window.addEventListener('resize', () => {{
        const w = container.clientWidth || window.innerWidth;
        const h_new = {height};
        camera.aspect = w / h_new;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h_new);
      }});
    }}
  </script>
</body>
</html>"""
  return html


def plot_3d_anatomical_scene(
    fluxes: Optional[Dict[str, float]] = None,
    is_starved: bool = False,
    is_parasitic: bool = False,
    show_soil_cutaway: bool = True,
) -> Any:
  """Renders an interactive Plotly 3D vector anatomical scene of the rice-rhizosphere twin.

  Features:
  - Parametric soil cylinder (go.Mesh3d) with 90° cutaway window when
  show_soil_cutaway=True.
  - Above-ground rice shoot canopy with 8 arching leaves (go.Scatter3d).
  - Subterranean root network with seminal and crown branching roots
  (go.Scatter3d).
  - OsSWEET11 transporter diamond nodes localized on root elongation zones.
  - Azospirillum brasilense bacterial clusters (cyan mutualistic vs crimson
  parasitic).
  - 3D flux cones for photon influx, SWEET11 sucrose exudation, and fixed NH4+
  influx.

  Parameters
  ----------
  fluxes : Optional[Dict[str, float]]
      Metabolic flux dictionary mapping reaction IDs to flux values.
  is_starved : bool
      Lethal host carbon starvation flag.
  is_parasitic : bool
      Bacterial parasitism / nitrogenase shutoff flag.
  show_soil_cutaway : bool
      Whether to render the 90-degree cutaway opening exposing root core.

  Returns
  -------
  go.Figure
      Interactive 3D Plotly figure.
  """
  if go is None:
    return None

  if fluxes is None:
    fluxes = {}

  # Robust flux value extraction and clamping
  photon_flux = max(
      0.0, float(fluxes.get("EX_photon_r", 100.0 if not is_starved else 0.0))
  )
  sweet11_flux = max(
      0.0, float(fluxes.get("SWEET11_export", 1.842 if not is_starved else 0.0))
  )
  nitrogenase_flux = max(
      0.0,
      float(
          fluxes.get(
              "NITROGENASE_ab",
              1.150 if not is_parasitic and not is_starved else 0.0,
          )
      ),
  )
  nh4_flux = max(
      0.0, float(fluxes.get("NH4_excretion_ab", nitrogenase_flux * 2.0))
  )
  biomass_rice = max(
      0.0, float(fluxes.get("BIOMASS_rice", 0.125 if not is_starved else 0.0))
  )
  atpm_r = max(0.0, float(fluxes.get("ATPM_r", 2.50)))

  fig = go.Figure()

  # -------------------------------------------------------------------------
  # 1. Parametric Soil Cylinder (go.Mesh3d)
  # -------------------------------------------------------------------------
  r_soil = 3.2
  z_min, z_max = -6.0, 0.0
  n_theta, n_z = 24, 8
  theta_start = np.pi * 0.5 if show_soil_cutaway else 0.0
  theta_end = np.pi * 2.0 if show_soil_cutaway else 2.0 * np.pi

  thetas = np.linspace(theta_start, theta_end, n_theta)
  zs = np.linspace(z_min, z_max, n_z)

  verts = []
  for zi in zs:
    for ti in thetas:
      verts.append([r_soil * np.cos(ti), r_soil * np.sin(ti), zi])
  wall_count = len(verts)

  for zi in zs:
    verts.append([0.0, 0.0, zi])
  axis_start = wall_count

  i_idx, j_idx, k_idx = [], [], []

  # Outer curved cylinder wall quads -> triangles
  for iz in range(n_z - 1):
    for it in range(n_theta - 1):
      p1 = iz * n_theta + it
      p2 = iz * n_theta + it + 1
      p3 = (iz + 1) * n_theta + it + 1
      p4 = (iz + 1) * n_theta + it
      i_idx.extend([p1, p1])
      j_idx.extend([p2, p3])
      k_idx.extend([p3, p4])

  # Top ground lid fan
  top_axis = axis_start + (n_z - 1)
  top_offset = (n_z - 1) * n_theta
  for it in range(n_theta - 1):
    i_idx.append(top_axis)
    j_idx.append(top_offset + it)
    k_idx.append(top_offset + it + 1)

  # Bottom floor lid fan
  bot_axis = axis_start
  for it in range(n_theta - 1):
    i_idx.append(bot_axis)
    j_idx.append(it + 1)
    k_idx.append(it)

  # Radial interior cutaway walls (only if cutaway active)
  if show_soil_cutaway:
    # Wall at theta_start (it = 0)
    for iz in range(n_z - 1):
      a1 = axis_start + iz
      a2 = axis_start + iz + 1
      w1 = iz * n_theta
      w2 = (iz + 1) * n_theta
      i_idx.extend([a1, a1])
      j_idx.extend([w1, w2])
      k_idx.extend([w2, a2])
    # Wall at theta_end (it = n_theta - 1)
    for iz in range(n_z - 1):
      a1 = axis_start + iz
      a2 = axis_start + iz + 1
      w1 = iz * n_theta + (n_theta - 1)
      w2 = (iz + 1) * n_theta + (n_theta - 1)
      i_idx.extend([a1, a1])
      j_idx.extend([a2, w2])
      k_idx.extend([w2, w1])

  verts = np.array(verts)
  fig.add_trace(
      go.Mesh3d(
          x=verts[:, 0],
          y=verts[:, 1],
          z=verts[:, 2],
          i=i_idx,
          j=j_idx,
          k=k_idx,
          opacity=0.20,
          color="#78350f",
          name="Soil Rhizosphere Volume",
          hoverinfo="skip",
      )
  )

  # -------------------------------------------------------------------------
  # 2. Rice Shoot Canopy (go.Scatter3d)
  # -------------------------------------------------------------------------
  canopy_color = (
      "#582d0e"
      if is_starved
      else ("#22c55e" if biomass_rice >= 0.05 else "#eab308")
  )
  cx, cy, cz = [], [], []

  # Central tiller culm
  stem_z = np.linspace(0, 4.5, 12)
  cx.extend([0.0] * 12 + [None])
  cy.extend([0.0] * 12 + [None])
  cz.extend(stem_z.tolist() + [None])

  # 8 Arching vegetative leaves
  n_leaves = 8
  for k in range(n_leaves):
    phi = 2 * np.pi * k / n_leaves + 0.15 * np.sin(k)
    s = np.linspace(0, 1, 16)
    z_base = 0.4 + 2.2 * (k / n_leaves)
    r = 0.15 + 2.6 * (s**1.1)
    z = z_base + 2.8 * s - 3.6 * (s**2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    cx.extend(x.tolist() + [None])
    cy.extend(y.tolist() + [None])
    cz.extend(z.tolist() + [None])

  state_str = (
      "Necrotic Starvation"
      if is_starved
      else (
          "Vibrant Vegetative"
          if biomass_rice >= 0.05
          else "Mild Light/Nitrogen Limitation"
      )
  )
  fig.add_trace(
      go.Scatter3d(
          x=cx,
          y=cy,
          z=cz,
          mode="lines+markers",
          line=dict(color=canopy_color, width=4),
          marker=dict(size=3, color=canopy_color),
          name="Rice Shoot Canopy",
          customdata=[[biomass_rice, state_str]] * len(cx),
          hovertemplate=(
              "<b>Rice Shoot Canopy</b><br>Biomass Rate: %{customdata[0]:.4f}"
              " h⁻¹<br>Status: %{customdata[1]}<extra></extra>"
          ),
      )
  )

  # -------------------------------------------------------------------------
  # 3. Branching Root Network (go.Scatter3d)
  # -------------------------------------------------------------------------
  rx, ry, rz = [], [], []
  sweet_pts = []
  bact_pts = []

  # Primary seminal root
  z_sem = np.linspace(0, -5.5, 18)
  x_sem = 0.04 * np.sin(z_sem * 1.5)
  y_sem = 0.04 * np.cos(z_sem * 1.5)
  rx.extend(x_sem.tolist() + [None])
  ry.extend(y_sem.tolist() + [None])
  rz.extend(z_sem.tolist() + [None])
  sweet_pts.append((float(x_sem[7]), float(y_sem[7]), float(z_sem[7])))

  # 6 Crown adventitious roots & lateral branches
  n_crown = 6
  for i in range(n_crown):
    theta = 2 * np.pi * i / n_crown + 0.05 * np.sin(i * 1.7)
    t = np.linspace(0, 1, 15)
    depth = 4.2 + 0.6 * ((i * 3) % 4) / 3.0
    spread = 1.3 + 0.4 * ((i * 5) % 3) / 2.0
    cr_z = -depth * t
    cr_r = spread * (t**0.85)
    cr_x = cr_r * np.cos(theta)
    cr_y = cr_r * np.sin(theta)
    rx.extend(cr_x.tolist() + [None])
    ry.extend(cr_y.tolist() + [None])
    rz.extend(cr_z.tolist() + [None])

    # SWEET11 node on elongation zone of crown root
    idx_sw = 6
    sweet_pts.append(
        (float(cr_x[idx_sw]), float(cr_y[idx_sw]), float(cr_z[idx_sw]))
    )

    # Lateral root branches
    for j_lat in [5, 10]:
      lat_t = np.linspace(0, 1, 8)
      lat_theta = theta + (0.8 if (i + j_lat) % 2 == 0 else -0.8)
      ox, oy, oz = cr_x[j_lat], cr_y[j_lat], cr_z[j_lat]
      lx = ox + 0.8 * lat_t * np.cos(lat_theta)
      ly = oy + 0.8 * lat_t * np.sin(lat_theta)
      lz = oz - 1.0 * (lat_t**1.1)
      rx.extend(lx.tolist() + [None])
      ry.extend(ly.tolist() + [None])
      rz.extend(lz.tolist() + [None])

      bact_pts.append((
          float(lx[-1] + 0.08 * np.sin(i * 3.1)),
          float(ly[-1] + 0.08 * np.cos(i * 2.7)),
          float(lz[-1] - 0.04),
      ))

  fig.add_trace(
      go.Scatter3d(
          x=rx,
          y=ry,
          z=rz,
          mode="lines",
          line=dict(color="#d97706", width=3),
          name="Rice Root Network",
          customdata=[atpm_r] * len(rx),
          hovertemplate=(
              "<b>Rice Root Network</b><br>Maintenance ATP: %{customdata:.2f}"
              " mmol/gDW/h<br>Zone: Cortex / Vascular Bundle<extra></extra>"
          ),
      )
  )

  # -------------------------------------------------------------------------
  # 4. OsSWEET11 Transporter Nodes (go.Scatter3d)
  # -------------------------------------------------------------------------
  sx = [p[0] for p in sweet_pts]
  sy = [p[1] for p in sweet_pts]
  sz = [p[2] for p in sweet_pts]
  sweet_color = "#9ca3af" if sweet11_flux < 1e-4 else "#0284c7"
  sweet_status = (
      "Knockout / Inactive" if sweet11_flux < 1e-4 else "Active Sucrose Export"
  )
  fig.add_trace(
      go.Scatter3d(
          x=sx,
          y=sy,
          z=sz,
          mode="markers",
          marker=dict(
              size=7,
              color=sweet_color,
              symbol="diamond",
              line=dict(color="#38bdf8", width=1),
          ),
          name="OsSWEET11 Transporters",
          customdata=[[sweet11_flux, sweet_status]] * len(sx),
          hovertemplate=(
              "<b>OsSWEET11 Transporter Node</b><br>Sucrose Flux:"
              " %{customdata[0]:.3f} mmol/gDW/h<br>Status:"
              " %{customdata[1]}<extra></extra>"
          ),
      )
  )

  # -------------------------------------------------------------------------
  # 5. Azospirillum brasilense Bacterial Colonies (go.Scatter3d)
  # -------------------------------------------------------------------------
  bx = [p[0] for p in bact_pts]
  by = [p[1] for p in bact_pts]
  bz = [p[2] for p in bact_pts]
  bact_color = "#ef4444" if is_parasitic else "#06b6d4"
  bact_status = (
      "PARASITIC CARBON DRAIN" if is_parasitic else "MUTUALISTIC N2 FIXATION"
  )
  fig.add_trace(
      go.Scatter3d(
          x=bx,
          y=by,
          z=bz,
          mode="markers",
          marker=dict(size=6, color=bact_color, symbol="circle", opacity=0.9),
          name="A. brasilense Colonies",
          customdata=[[nitrogenase_flux, nh4_flux, bact_status]] * len(bx),
          hovertemplate=(
              "<b>A. brasilense Bacterial Colony</b><br>Nitrogenase Flux:"
              " %{customdata[0]:.3f} mmol/gDW/h<br>NH4+ Excretion:"
              " %{customdata[1]:.3f} mmol/gDW/h<br>Status:"
              " %{customdata[2]}<extra></extra>"
          ),
      )
  )

  # -------------------------------------------------------------------------
  # 6. 3D Flux Vectors (go.Cone)
  # -------------------------------------------------------------------------
  # Photon cones (sky -> canopy)
  if photon_flux > 1e-4:
    n_p = 6
    p_thetas = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    px = 1.4 * np.cos(p_thetas)
    py = 1.4 * np.sin(p_thetas)
    pz = np.full(n_p, 5.0)
    pu = np.zeros(n_p)
    pv = np.zeros(n_p)
    pw = np.full(n_p, -1.2 * min(2.0, max(0.4, photon_flux / 50.0)))
    fig.add_trace(
        go.Cone(
            x=px,
            y=py,
            z=pz,
            u=pu,
            v=pv,
            w=pw,
            sizemode="scaled",
            sizeref=0.35,
            anchor="tail",
            colorscale=[[0, "#fef08a"], [1, "#f59e0b"]],
            showscale=False,
            name="Solar Photon Influx",
            customdata=[photon_flux] * n_p,
            hovertemplate=(
                "<b>Solar Photon Flux</b>: %{customdata:.1f}"
                " mmol/gDW/h<extra></extra>"
            ),
        )
    )

  # SWEET11 cones (root -> rhizosphere)
  if sweet11_flux > 1e-4:
    cone_sx, cone_sy, cone_sz = [], [], []
    cone_su, cone_sv, cone_sw = [], [], []
    scale_sw = min(1.5, max(0.3, sweet11_flux / 1.5))
    for p in sweet_pts[:4]:
      cone_sx.append(p[0])
      cone_sy.append(p[1])
      cone_sz.append(p[2])
      r_dist = np.sqrt(p[0] ** 2 + p[1] ** 2) + 1e-5
      cone_su.append(0.5 * scale_sw * (p[0] / r_dist))
      cone_sv.append(0.5 * scale_sw * (p[1] / r_dist))
      cone_sw.append(-0.5 * scale_sw)
    fig.add_trace(
        go.Cone(
            x=cone_sx,
            y=cone_sy,
            z=cone_sz,
            u=cone_su,
            v=cone_sv,
            w=cone_sw,
            sizemode="scaled",
            sizeref=0.3,
            anchor="tail",
            colorscale=[[0, "#7dd3fc"], [1, "#0284c7"]],
            showscale=False,
            name="SWEET11 Sucrose Flux",
            customdata=[sweet11_flux] * len(cone_sx),
            hovertemplate=(
                "<b>SWEET11 Sucrose Flux</b>: %{customdata:.3f}"
                " mmol/gDW/h<extra></extra>"
            ),
        )
    )

  # Ammonium cones (bacteria -> root xylem)
  if nh4_flux > 1e-4 and not is_parasitic:
    cone_bx, cone_by, cone_bz = [], [], []
    cone_bu, cone_bv, cone_bw = [], [], []
    scale_nh4 = min(1.5, max(0.3, nh4_flux / 2.0))
    for p in bact_pts[:5]:
      cone_bx.append(p[0])
      cone_by.append(p[1])
      cone_bz.append(p[2])
      r_dist = np.sqrt(p[0] ** 2 + p[1] ** 2) + 1e-5
      cone_bu.append(-0.5 * scale_nh4 * (p[0] / r_dist))
      cone_bv.append(-0.5 * scale_nh4 * (p[1] / r_dist))
      cone_bw.append(0.3 * scale_nh4)
    fig.add_trace(
        go.Cone(
            x=cone_bx,
            y=cone_by,
            z=cone_bz,
            u=cone_bu,
            v=cone_bv,
            w=cone_bw,
            sizemode="scaled",
            sizeref=0.3,
            anchor="tail",
            colorscale=[[0, "#f472b6"], [1, "#a855f7"]],
            showscale=False,
            name="Diazotrophic NH4+ Flux",
            customdata=[nh4_flux] * len(cone_bx),
            hovertemplate=(
                "<b>Diazotrophic NH4+ Flux</b>: %{customdata:.3f}"
                " mmol/gDW/h<extra></extra>"
            ),
        )
    )

  # -------------------------------------------------------------------------
  # 7. Scene Layout & Dark Theme
  # -------------------------------------------------------------------------
  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="rgba(11, 15, 25, 1)",
      plot_bgcolor="rgba(11, 15, 25, 1)",
      margin=dict(l=0, r=0, b=0, t=30),
      legend=dict(
          x=0.02,
          y=0.98,
          bgcolor="rgba(17, 24, 39, 0.85)",
          bordercolor="#374151",
          borderwidth=1,
          font=dict(color="#e5e7eb", size=10),
      ),
      scene=dict(
          aspectmode="manual",
          aspectratio=dict(x=1.0, y=1.0, z=1.8),
          xaxis=dict(
              title="X (mm)",
              backgroundcolor="rgba(17, 24, 39, 0.5)",
              gridcolor="rgba(255, 255, 255, 0.08)",
              zerolinecolor="rgba(255, 255, 255, 0.2)",
          ),
          yaxis=dict(
              title="Y (mm)",
              backgroundcolor="rgba(17, 24, 39, 0.5)",
              gridcolor="rgba(255, 255, 255, 0.08)",
              zerolinecolor="rgba(255, 255, 255, 0.2)",
          ),
          zaxis=dict(
              title="Depth Z (mm)",
              backgroundcolor="rgba(17, 24, 39, 0.5)",
              gridcolor="rgba(255, 255, 255, 0.08)",
              zerolinecolor="rgba(255, 255, 255, 0.2)",
          ),
          camera=dict(
              eye=dict(x=1.8, y=1.8, z=1.2),
              center=dict(x=0.0, y=0.0, z=-0.1),
              up=dict(x=0.0, y=0.0, z=1.0),
          ),
      ),
  )

  return fig


# ==============================================================================
# Engine B: 3D Mathematical Phase Plane Surfaces
# ==============================================================================


def plot_3d_pareto_surface(
    phase_result: Any,
    current_genotype: Optional[str] = None,
    colorscale: str = "Viridis",
) -> Any:
  """Renders a rotatable 3D Pareto surface of rice biomass vs light & SWEET11.

  Visualizes:
  - 3D Surface: X=Light %, Y=SWEET11 Export, Z=Rice Biomass, colored by
  Nitrogenase.
  - Optimal Pareto Ridge: 3D spline tracing maximal biomass coordinates.
  - Starvation Cliff: 3D boundary line where cellular ATP maintenance fails.
  - Genotype Benchmark Markers: WT, OE, Knockdown, KO with active dynamic
  highlighting.

  Parameters
  ----------
  phase_result : Any
      PhasePlane2DResult dataclass or compatible dict.
  current_genotype : Optional[str]
      Name of currently active genotype to highlight on the manifold.
  colorscale : str
      Continuous Plotly colorscale for nitrogenase activity (default 'Viridis').

  Returns
  -------
  go.Figure
      Interactive 3D Plotly figure.
  """
  if go is None:
    return None
  if phase_result is None:
    return _create_empty_dark_figure("No Phase Plane Data Available")

  # Safe polymorphic attribute extraction
  X = getattr(phase_result, "X", None)
  if X is None and isinstance(phase_result, dict):
    X = phase_result.get("X", phase_result.get("X_light", None))
  Y = getattr(phase_result, "Y", None)
  if Y is None and isinstance(phase_result, dict):
    Y = phase_result.get("Y", phase_result.get("Y_sweet11", None))
  Z_biomass = getattr(phase_result, "Z_biomass_rice", None)
  if Z_biomass is None and isinstance(phase_result, dict):
    Z_biomass = phase_result.get(
        "Z_biomass_rice", phase_result.get("Z_rice_biomass", None)
    )
  Z_nif = getattr(phase_result, "Z_nitrogenase_ab", None)
  if Z_nif is None and isinstance(phase_result, dict):
    Z_nif = phase_result.get(
        "Z_nitrogenase_ab", phase_result.get("Z_nitrogenase", None)
    )

  if X is None or Y is None or Z_biomass is None:
    return _create_empty_dark_figure("Incomplete Phase Plane Grid Arrays")

  X = np.asarray(X, dtype=np.float64)
  Y = np.asarray(Y, dtype=np.float64)
  Z_biomass = np.nan_to_num(np.asarray(Z_biomass, dtype=np.float64), nan=0.0)
  Z_nif = np.nan_to_num(
      np.asarray(
          Z_nif if Z_nif is not None else Z_biomass, dtype=np.float64
      ),
      nan=0.0,
  )

  # 1D coordinate vectors
  x_vals = getattr(phase_result, "x_values", None)
  if x_vals is None:
    x_vals = X[0, :] if X.ndim == 2 else X
  y_vals = getattr(phase_result, "y_values", None)
  if y_vals is None:
    y_vals = Y[:, 0] if Y.ndim == 2 else Y

  x_vals = np.asarray(x_vals, dtype=np.float64)
  y_vals = np.asarray(y_vals, dtype=np.float64)

  fig = go.Figure()

  # --------------------------------------------------------------------------
  # Trace 1: 3D Biomass Surface colored by Nitrogenase
  # --------------------------------------------------------------------------
  cmax_val = max(1e-4, float(np.max(Z_nif)))
  fig.add_trace(
      go.Surface(
          x=X,
          y=Y,
          z=Z_biomass,
          surfacecolor=Z_nif,
          colorscale=colorscale,
          cmin=0.0,
          cmax=cmax_val,
          colorbar=dict(
              title=dict(
                  text="Nitrogenase Activity<br>(mmol NH₄⁺/gDW/h)",
                  font=dict(color="#f8fafc", size=12),
                  side="right",
              ),
              tickfont=dict(color="#94a3b8"),
              len=0.75,
              thickness=18,
              x=1.02,
          ),
          opacity=0.92,
          contours=dict(
              z=dict(
                  show=True,
                  usecolormap=True,
                  project_z=False,
                  highlightcolor="#38bdf8",
                  width=1.5,
              )
          ),
          lighting=dict(
              ambient=0.6, diffuse=0.8, specular=0.25, roughness=0.5
          ),
          hovertemplate=(
              "<b>Solar Irradiance</b>: %{x:.1f}%<br><b>SWEET11 Exudation</b>:"
              " %{y:.2f} mmol/gDW/h<br><b>Rice Biomass</b>: %{z:.4f}"
              " h⁻¹<br><b>Nitrogenase Activity</b>: %{surfacecolor:.4f} mmol"
              " NH₄⁺/gDW/h<extra></extra>"
          ),
          name="Biomass-Nitrogenase Manifold",
      )
  )

  # --------------------------------------------------------------------------
  # Trace 2: Optimal Pareto Ridge
  # --------------------------------------------------------------------------
  ridge_x = getattr(phase_result, "pareto_ridge_x", None)
  ridge_y = getattr(phase_result, "pareto_ridge_y", None)
  if ridge_x is not None and ridge_y is not None and len(ridge_x) > 0:
    rx = np.asarray(ridge_x, dtype=np.float64)
    ry = np.asarray(ridge_y, dtype=np.float64)
    rz = []
    for xv, yv in zip(rx, ry):
      ix = int(np.argmin(np.abs(x_vals - xv)))
      iy = int(np.argmin(np.abs(y_vals - yv)))
      rz.append(float(Z_biomass[iy, ix]) + 0.004)  # Elevation offset

    fig.add_trace(
        go.Scatter3d(
            x=rx,
            y=ry,
            z=rz,
            mode="lines+markers",
            line=dict(color="#10b981", width=7),
            marker=dict(size=4, color="#10b981"),
            name="Optimal Pareto Ridge",
            hovertemplate=(
                "<b>Optimal Pareto Ridge</b><br>Light: %{x:.1f}%<br>Optimal"
                " SWEET11: %{y:.2f} mmol/gDW/h<br>Max Biomass: %{z:.4f}"
                " h⁻¹<extra></extra>"
            ),
        )
    )

  # --------------------------------------------------------------------------
  # Trace 3: Starvation Cliff Boundary
  # --------------------------------------------------------------------------
  cliff_x = getattr(phase_result, "cliff_x", None)
  cliff_y = getattr(phase_result, "cliff_y", None)
  if cliff_x is not None and cliff_y is not None and len(cliff_x) >= 2:
    cx = np.asarray(cliff_x, dtype=np.float64)
    cy = np.asarray(cliff_y, dtype=np.float64)
    cz = []
    for xv, yv in zip(cx, cy):
      ix = int(np.argmin(np.abs(x_vals - xv)))
      iy = int(np.argmin(np.abs(y_vals - yv)))
      cz.append(float(Z_biomass[iy, ix]) + 0.002)

    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="lines+markers",
            line=dict(color="#ef4444", width=5, dash="dash"),
            marker=dict(size=5, color="#ef4444", symbol="x"),
            name="Starvation Cliff (ATPM Failure)",
            hovertemplate=(
                "<b>Carbon Starvation Cliff</b><br>Light: %{x:.1f}%<br>SWEET11"
                " Threshold: %{y:.2f} mmol/gDW/h<br>Status: Lethal Maintenance"
                " Deficit<extra></extra>"
            ),
        )
    )

  # --------------------------------------------------------------------------
  # Trace 4+: Genotype Benchmark Markers with Dynamic Highlighting
  # --------------------------------------------------------------------------
  genotypes_pts = [
      ("WT", "OsSWEET11_WT", 100.0, 4.0, "#06b6d4", "diamond"),
      ("OE", "OsSWEET11_OE", 70.0, 8.0, "#f59e0b", "cross"),
      ("Knockdown", "OsSWEET11_Knockdown", 70.0, 2.0, "#f97316", "square"),
      ("KO", "Ossweet11_KO", 50.0, 0.0, "#94a3b8", "circle"),
  ]

  curr_norm = _normalize_genotype(current_genotype)

  for short_name, full_name, gx, gy, color, sym in genotypes_pts:
    ix = int(np.argmin(np.abs(x_vals - gx)))
    iy = int(np.argmin(np.abs(y_vals - gy)))
    gz = float(Z_biomass[iy, ix]) + 0.008

    is_active = (
        (
            curr_norm in _normalize_genotype(short_name)
            or curr_norm in _normalize_genotype(full_name)
        )
        if curr_norm
        else False
    )

    m_size = 14 if is_active else 8
    m_border = (
        dict(color="#ffffff", width=3)
        if is_active
        else dict(color="#000000", width=1)
    )
    label_text = f"★ {short_name} (ACTIVE)" if is_active else short_name

    fig.add_trace(
        go.Scatter3d(
            x=[gx],
            y=[gy],
            z=[gz],
            mode="markers+text",
            text=[label_text],
            textposition="top center" if is_active else "bottom right",
            textfont=dict(
                color="#ffffff" if is_active else "#cbd5e1",
                size=12 if is_active else 10,
            ),
            marker=dict(size=m_size, symbol=sym, color=color, line=m_border),
            name=f"Genotype: {short_name}" + (" [ACTIVE]" if is_active else ""),
            hovertemplate=(
                f"<b>Genotype</b>: {full_name}<br>Light: %{{x:.1f}}%<br>SWEET11"
                " Export: %{y:.2f} mmol/gDW/h<br>Biomass: %{z:.4f}"
                " h⁻¹<extra></extra>"
            ),
        )
    )

  # --------------------------------------------------------------------------
  # Scene Layout & Dark Scientific Styling
  # --------------------------------------------------------------------------
  param_x_title = getattr(phase_result, "param_x_name", "Solar Irradiance")
  param_y_title = getattr(phase_result, "param_y_name", "SWEET11 Exudation")

  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="#0b0f19",
      plot_bgcolor="#0b0f19",
      scene=dict(
          xaxis=dict(
              title=f"{param_x_title} (%)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          yaxis=dict(
              title=f"{param_y_title} (mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          zaxis=dict(
              title="Rice Biomass (h⁻¹)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          camera=dict(
              eye=dict(x=1.75, y=-1.75, z=1.25),
              up=dict(x=0, y=0, z=1),
          ),
          aspectmode="cube",
      ),
      legend=dict(
          x=0.02,
          y=0.98,
          bgcolor="rgba(15, 23, 42, 0.8)",
          bordercolor="#334155",
          borderwidth=1,
          font=dict(color="#f8fafc", size=11),
      ),
      margin=dict(l=10, r=10, b=10, t=50),
      title=dict(
          text="3D Phenotypic Phase Plane: Multi-Objective Pareto Surface",
          font=dict(color="#f8fafc", size=16),
      ),
  )

  return fig


def plot_3d_ammonium_collapse_surface(
    bifurcation_result: Any,
) -> Any:
  """Renders the 3D Soil Ammonium Collapse Landscape with Z=0 Neutral Plane.

  Visualizes:
  - 3D Bifurcation Surface: X=Soil NH4+, Y=SWEET11, Z=Net Host Benefit (NHBI).
  - Diverging Colormap ('RdBu_r' with cmid=0.0): Mutualistic dome vs parasitic
  valley.
  - Z = 0 Reference Plane: Semi-transparent plane dividing mutualism from
  parasitism.
  - Critical Tipping Point Line: Tracing [NH4+]crit boundary contour.
  - Callout Marker: Pinpointing exact critical chemical fertilizer displacement
  point.

  Parameters
  ----------
  bifurcation_result : Any
      AmmoniumBifurcationResult dataclass or compatible dict.

  Returns
  -------
  go.Figure
      Interactive 3D Plotly figure.
  """
  if go is None:
    return None
  if bifurcation_result is None:
    return _create_empty_dark_figure("No Bifurcation Data Available")

  # Safe polymorphic extraction
  X_nh4 = getattr(bifurcation_result, "X_nh4", None)
  if X_nh4 is None and isinstance(bifurcation_result, dict):
    X_nh4 = bifurcation_result.get("X_nh4", bifurcation_result.get("X", None))
  Y_sw = getattr(bifurcation_result, "Y_sweet11", None)
  if Y_sw is None and isinstance(bifurcation_result, dict):
    Y_sw = bifurcation_result.get("Y_sweet11", bifurcation_result.get("Y", None))
  Z_nhbi = getattr(bifurcation_result, "Z_nhbi", None)
  if Z_nhbi is None and isinstance(bifurcation_result, dict):
    Z_nhbi = bifurcation_result.get("Z_nhbi", None)

  crit_nh4 = float(
      getattr(bifurcation_result, "critical_nh4_threshold", 5.0)
      if not isinstance(bifurcation_result, dict)
      else bifurcation_result.get("critical_nh4_threshold", 5.0)
  )

  if X_nh4 is None or Y_sw is None or Z_nhbi is None:
    return _create_empty_dark_figure("Incomplete Ammonium Bifurcation Arrays")

  X_nh4 = np.asarray(X_nh4, dtype=np.float64)
  Y_sw = np.asarray(Y_sw, dtype=np.float64)
  Z_nhbi = np.nan_to_num(np.asarray(Z_nhbi, dtype=np.float64), nan=0.0)

  fig = go.Figure()

  # Dynamic state labels for hovercard
  state_labels = np.where(
      Z_nhbi >= 0,
      "Mutualistic Symbiosis (+Growth Benefit)",
      "Parasitic Carbon Drain (-Biomass Loss)",
  )

  # --------------------------------------------------------------------------
  # Trace 1: 3D NHBI Collapse Surface
  # --------------------------------------------------------------------------
  fig.add_trace(
      go.Surface(
          x=X_nh4,
          y=Y_sw,
          z=Z_nhbi,
          colorscale="RdBu_r",
          cmid=0.0,
          colorbar=dict(
              title=dict(
                  text="Net Host Benefit (NHBI)<br>(Δ Biomass h⁻¹)",
                  font=dict(color="#f8fafc", size=12),
                  side="right",
              ),
              tickfont=dict(color="#94a3b8"),
              len=0.75,
              thickness=18,
              x=1.02,
          ),
          opacity=0.92,
          customdata=state_labels,
          hovertemplate=(
              "<b>Soil NH₄⁺</b>: %{x:.2f} mmol/gDW/h<br><b>SWEET11 Export</b>:"
              " %{y:.2f} mmol/gDW/h<br><b>Net Host Benefit (NHBI)</b>: %{z:.4f}"
              " h⁻¹<br><b>Regime</b>: %{customdata}<extra></extra>"
          ),
          contours=dict(
              z=dict(
                  show=True,
                  usecolormap=True,
                  project_z=False,
                  highlightcolor="#f59e0b",
                  width=2,
              )
          ),
          lighting=dict(ambient=0.6, diffuse=0.8, specular=0.25, roughness=0.5),
          name="Ammonium Collapse Surface",
      )
  )

  # --------------------------------------------------------------------------
  # Trace 2: Neutral Benefit Reference Plane (Z = 0)
  # --------------------------------------------------------------------------
  z_zero = np.zeros_like(Z_nhbi)
  fig.add_trace(
      go.Surface(
          x=X_nh4,
          y=Y_sw,
          z=z_zero,
          showscale=False,
          opacity=0.35,
          colorscale=[
              [0, "rgba(148, 163, 184, 0.4)"],
              [1, "rgba(148, 163, 184, 0.4)"],
          ],
          name="Neutral Benefit Plane (Z = 0)",
          hovertemplate=(
              "<b>Neutral Threshold Plane (Z = 0)</b><br>Boundary: Mutualism (Z"
              " > 0) vs Parasitism (Z < 0)<extra></extra>"
          ),
          contours=dict(
              x=dict(show=True, color="#475569", width=1),
              y=dict(show=True, color="#475569", width=1),
          ),
      )
  )

  # --------------------------------------------------------------------------
  # Trace 3: Tipping Point Bifurcation Line
  # --------------------------------------------------------------------------
  bif_contour = getattr(bifurcation_result, "bifurcation_contour", None)
  if isinstance(bifurcation_result, dict):
    bif_contour = bifurcation_result.get("bifurcation_contour", None)

  if (
      isinstance(bif_contour, pd.DataFrame)
      and len(bif_contour) > 0
      and "critical_nh4" in bif_contour
  ):
    bx = bif_contour["critical_nh4"].values
    by = bif_contour["sweet11_export"].values
    bz = np.zeros_like(bx) + 0.001
  else:
    y_min = float(np.min(Y_sw))
    y_max = float(np.max(Y_sw))
    bx = np.array([crit_nh4, crit_nh4], dtype=np.float64)
    by = np.array([y_min, y_max], dtype=np.float64)
    bz = np.array([0.001, 0.001], dtype=np.float64)

  fig.add_trace(
      go.Scatter3d(
          x=bx,
          y=by,
          z=bz,
          mode="lines+markers",
          line=dict(color="#f59e0b", width=7),
          marker=dict(size=4, color="#f59e0b"),
          name=f"Tipping Boundary ([NH₄⁺]crit = {crit_nh4:.2f})",
          hovertemplate=(
              "<b>Ecological Tipping Point</b><br>Critical Soil NH₄⁺: %{x:.2f}"
              " mmol/gDW/h<br>SWEET11 Export: %{y:.2f} mmol/gDW/h<br>Net Host"
              " Benefit: 0.0000 h⁻¹<extra></extra>"
          ),
      )
  )

  # --------------------------------------------------------------------------
  # Trace 4: Tipping Point Highlight Marker
  # --------------------------------------------------------------------------
  mid_y = float(np.mean(Y_sw))
  fig.add_trace(
      go.Scatter3d(
          x=[crit_nh4],
          y=[mid_y],
          z=[0.002],
          mode="markers+text",
          text=[f"[NH₄⁺]crit = {crit_nh4:.2f}"],
          textposition="top center",
          textfont=dict(color="#f59e0b", size=12),
          marker=dict(
              size=10,
              symbol="diamond",
              color="#f59e0b",
              line=dict(color="#ffffff", width=2),
          ),
          name="Critical Threshold Marker",
          hovertemplate=(
              f"<b>Bifurcation Point</b>: [NH₄⁺]crit = {crit_nh4:.2f}"
              " mmol/gDW/h<br>Synthetic fertilizer threshold where diazotroph"
              " shifts to parasite.<extra></extra>"
          ),
      )
  )

  # --------------------------------------------------------------------------
  # Scene Layout & Dark Scientific Styling
  # --------------------------------------------------------------------------
  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="#0b0f19",
      plot_bgcolor="#0b0f19",
      scene=dict(
          xaxis=dict(
              title="Soil NH₄⁺ (mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          yaxis=dict(
              title="SWEET11 Export (mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          zaxis=dict(
              title="Net Host Benefit (NHBI, h⁻¹)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          camera=dict(
              eye=dict(x=1.8, y=-1.8, z=1.2),
              up=dict(x=0, y=0, z=1),
          ),
          aspectmode="cube",
      ),
      legend=dict(
          x=0.02,
          y=0.98,
          bgcolor="rgba(15, 23, 42, 0.8)",
          bordercolor="#334155",
          borderwidth=1,
          font=dict(color="#f8fafc", size=11),
      ),
      margin=dict(l=10, r=10, b=10, t=50),
      title=dict(
          text=(
              "3D Soil Ammonium Collapse Landscape | [NH₄⁺]crit ="
              f" {crit_nh4:.2f} mmol/gDW/h"
          ),
          font=dict(color="#f8fafc", size=16),
      ),
  )

  return fig


def plot_3d_flux_space_trajectory(
    exp1_result: Any,
) -> Any:
  """Renders the 3D metabolic state trajectory across 4 Philippine seasons.

  Visualizes:
  - 3D Coordinates: X=SUC_SYN_r, Y=SWEET11_export, Z=NITROGENASE_ab.
  - Closed Seasonal Loop: Peak Dry -> Monsoon -> Rainy -> Typhoon -> Peak Dry.
  - Milestone Markers: Color-coded glyphs (Gold, Aqua, Blue, Crimson).
  - Directional Cones: Indicating seasonal progression and metabolic hysteresis.
  - Continuous Spline: High-resolution continuous light response curve
  (10-120%).

  Parameters
  ----------
  exp1_result : Any
      Experiment1Result dataclass or compatible dict.

  Returns
  -------
  go.Figure
      Interactive 3D Plotly figure.
  """
  if go is None:
    return None
  if exp1_result is None:
    return _create_empty_dark_figure("No Seasonal Trajectory Data Available")

  # Safe extraction of regimes dictionary
  regimes_dict = getattr(exp1_result, "seasonal_regimes", None)
  if regimes_dict is None and isinstance(exp1_result, dict):
    regimes_dict = exp1_result.get("seasonal_regimes", exp1_result)
  if not isinstance(regimes_dict, dict):
    return _create_empty_dark_figure("Missing Seasonal Regimes Dictionary")

  # Canonical 4 Philippine Regimes in chronological sequence
  CANONICAL_SEASONS = [
      (
          "Peak Dry (100% Sun)",
          [
              "100% Peak Dry Season Control",
              "100% Peak Dry",
              "Peak Dry",
              "100%",
          ],
          "#facc15",
          "circle",
      ),
      (
          "Monsoon (70% Sun)",
          [
              "70% Intermittent Monsoon Cloud Cover",
              "70% Monsoon",
              "Monsoon",
              "70%",
          ],
          "#2dd4bf",
          "diamond",
      ),
      (
          "Rainy Season (50% Sun)",
          [
              "50% Extended Rainy Season Overcast",
              "50% Overcast",
              "Rainy",
              "50%",
          ],
          "#0ea5e9",
          "square",
      ),
      (
          "Typhoon (30% Sun)",
          ["30% Active Typhoon Passage", "30% Typhoon", "Typhoon", "30%"],
          "#ef4444",
          "cross",
      ),
  ]

  regime_points = []
  for label, match_keys, color, symbol in CANONICAL_SEASONS:
    found_sol = None
    for k, sol in regimes_dict.items():
      if any(mk.lower() in k.lower() for mk in match_keys):
        found_sol = sol
        break

    if found_sol is not None:
      suc = (
          getattr(found_sol, "suc_syn_r", 0.0)
          if not isinstance(found_sol, dict)
          else found_sol.get("suc_syn_r", 0.0)
      )
      sw = (
          getattr(found_sol, "sweet11_export", 0.0)
          if not isinstance(found_sol, dict)
          else found_sol.get("sweet11_export", 0.0)
      )
      nif = (
          getattr(found_sol, "nitrogenase_ab", 0.0)
          if not isinstance(found_sol, dict)
          else found_sol.get("nitrogenase_ab", 0.0)
      )
      bio = (
          getattr(found_sol, "biomass_rice", 0.0)
          if not isinstance(found_sol, dict)
          else found_sol.get("biomass_rice", 0.0)
      )
      photon = (
          getattr(found_sol, "ex_photon_r", 0.0)
          if not isinstance(found_sol, dict)
          else found_sol.get("ex_photon_r", 0.0)
      )
      starved = (
          getattr(found_sol, "is_starved", False)
          if not isinstance(found_sol, dict)
          else found_sol.get("is_starved", False)
      )

      regime_points.append({
          "label": label,
          "x": float(suc),
          "y": float(sw),
          "z": float(nif),
          "biomass": float(bio),
          "photon": float(photon),
          "starved": bool(starved),
          "color": color,
          "symbol": symbol,
      })

  # Fallback if canonical keys were not matched
  if len(regime_points) < 2:
    colors_palette = ["#facc15", "#2dd4bf", "#0ea5e9", "#ef4444", "#a855f7"]
    symbols_palette = ["circle", "diamond", "square", "cross", "triangle-up"]
    seen_keys = set()
    for idx, (k, sol) in enumerate(regimes_dict.items()):
      norm_k = k.strip().lower()
      if norm_k in seen_keys:
        continue
      seen_keys.add(norm_k)

      suc = (
          getattr(sol, "suc_syn_r", 0.0)
          if not isinstance(sol, dict)
          else sol.get("suc_syn_r", 0.0)
      )
      sw = (
          getattr(sol, "sweet11_export", 0.0)
          if not isinstance(sol, dict)
          else sol.get("sweet11_export", 0.0)
      )
      nif = (
          getattr(sol, "nitrogenase_ab", 0.0)
          if not isinstance(sol, dict)
          else sol.get("nitrogenase_ab", 0.0)
      )
      bio = (
          getattr(sol, "biomass_rice", 0.0)
          if not isinstance(sol, dict)
          else sol.get("biomass_rice", 0.0)
      )
      photon = (
          getattr(sol, "ex_photon_r", 0.0)
          if not isinstance(sol, dict)
          else sol.get("ex_photon_r", 0.0)
      )
      starved = (
          getattr(sol, "is_starved", False)
          if not isinstance(sol, dict)
          else sol.get("is_starved", False)
      )

      regime_points.append({
          "label": k,
          "x": float(suc),
          "y": float(sw),
          "z": float(nif),
          "biomass": float(bio),
          "photon": float(photon),
          "starved": bool(starved),
          "color": colors_palette[idx % len(colors_palette)],
          "symbol": symbols_palette[idx % len(symbols_palette)],
      })

  if not regime_points:
    return _create_empty_dark_figure("No Valid Regime Coordinates Available")

  fig = go.Figure()

  # --------------------------------------------------------------------------
  # Trace 1: Continuous Light Response Spline (Background Path)
  # --------------------------------------------------------------------------
  cont_traj = getattr(exp1_result, "continuous_trajectory", None)
  if isinstance(exp1_result, dict):
    cont_traj = exp1_result.get("continuous_trajectory", None)

  if cont_traj is not None and hasattr(cont_traj, "flux_trajectories"):
    trajs = cont_traj.flux_trajectories
    if (
        "SUC_SYN_r" in trajs
        and "SWEET11_export" in trajs
        and "NITROGENASE_ab" in trajs
    ):
      fig.add_trace(
          go.Scatter3d(
              x=trajs["SUC_SYN_r"],
              y=trajs["SWEET11_export"],
              z=trajs["NITROGENASE_ab"],
              mode="lines",
              line=dict(color="rgba(56, 189, 248, 0.35)", width=3, dash="dot"),
              name="Continuous Irradiance Path (10-120%)",
              hoverinfo="skip",
          )
      )

  # --------------------------------------------------------------------------
  # Trace 2: Closed Annual Loop Trajectory Line
  # --------------------------------------------------------------------------
  xs = [p["x"] for p in regime_points]
  ys = [p["y"] for p in regime_points]
  zs = [p["z"] for p in regime_points]

  # Close loop back to first seasonal point
  xs_closed = xs + [xs[0]]
  ys_closed = ys + [ys[0]]
  zs_closed = zs + [zs[0]]

  fig.add_trace(
      go.Scatter3d(
          x=xs_closed,
          y=ys_closed,
          z=zs_closed,
          mode="lines",
          line=dict(color="#38bdf8", width=6),
          name="Seasonal Trajectory Loop",
          hoverinfo="skip",
      )
  )

  # --------------------------------------------------------------------------
  # Trace 3: Directional Transition Vectors (Cones)
  # --------------------------------------------------------------------------
  max_span = max(
      max(xs) - min(xs) if len(xs) > 1 else 1.0,
      max(ys) - min(ys) if len(ys) > 1 else 1.0,
      max(zs) - min(zs) if len(zs) > 1 else 1.0,
      1.0,
  )

  for i in range(len(regime_points)):
    p1 = regime_points[i]
    p2 = regime_points[(i + 1) % len(regime_points)]
    mx = (p1["x"] + p2["x"]) / 2.0
    my = (p1["y"] + p2["y"]) / 2.0
    mz = (p1["z"] + p2["z"]) / 2.0
    vx = p2["x"] - p1["x"]
    vy = p2["y"] - p1["y"]
    vz = p2["z"] - p1["z"]
    norm = math.sqrt(vx**2 + vy**2 + vz**2)
    if norm > 1e-4:
      fig.add_trace(
          go.Cone(
              x=[mx],
              y=[my],
              z=[mz],
              u=[vx / norm],
              v=[vy / norm],
              w=[vz / norm],
              sizemode="absolute",
              sizeref=0.12 * max_span,
              showscale=False,
              colorscale=[[0, "#38bdf8"], [1, "#38bdf8"]],
              name="Seasonal Progression Vector",
              hoverinfo="skip",
          )
      )

  # --------------------------------------------------------------------------
  # Trace 4+: Milestone Markers for Seasons
  # --------------------------------------------------------------------------
  for p in regime_points:
    status_text = (
        "Lethal Starvation" if p["starved"] else "Metabolically Feasible"
    )
    fig.add_trace(
        go.Scatter3d(
            x=[p["x"]],
            y=[p["y"]],
            z=[p["z"]],
            mode="markers+text",
            text=[p["label"]],
            textposition="top center",
            textfont=dict(color="#f8fafc", size=11),
            marker=dict(
                size=12,
                symbol=p["symbol"],
                color=p["color"],
                line=dict(color="#ffffff", width=2),
            ),
            name=p["label"],
            hovertemplate=(
                f"<b>Regime</b>: {p['label']}<br><b>Sucrose Synthesis"
                " (SUC_SYN_r)</b>: %{x:.2f} mmol/gDW/h<br><b>SWEET11 Export</b>:"
                " %{y:.2f} mmol/gDW/h<br><b>Nitrogenase Activity</b>: %{z:.2f}"
                " mmol/gDW/h<br><b>Rice Biomass</b>:"
                f" {p['biomass']:.4f} h⁻¹<br><b>Photon Influx</b>:"
                f" {p['photon']:.1f} mmol/gDW/h<br><b>Physiological Status</b>:"
                f" {status_text}<extra></extra>"
            ),
        )
    )

  # --------------------------------------------------------------------------
  # Scene Layout & Dark Scientific Styling
  # --------------------------------------------------------------------------
  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="#0b0f19",
      plot_bgcolor="#0b0f19",
      scene=dict(
          xaxis=dict(
              title="Sucrose Synthesis (SUC_SYN_r, mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          yaxis=dict(
              title="SWEET11 Export (SWEET11_export, mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          zaxis=dict(
              title="Nitrogenase Fixation (NITROGENASE_ab, mmol/gDW/h)",
              backgroundcolor="#0f172a",
              gridcolor="#1e293b",
              zerolinecolor="#334155",
              showbackground=True,
          ),
          camera=dict(
              eye=dict(x=1.8, y=1.8, z=1.2),
              up=dict(x=0, y=0, z=1),
          ),
          aspectmode="cube",
      ),
      legend=dict(
          x=0.02,
          y=0.98,
          bgcolor="rgba(15, 23, 42, 0.8)",
          bordercolor="#334155",
          borderwidth=1,
          font=dict(color="#f8fafc", size=11),
      ),
      margin=dict(l=10, r=10, b=10, t=50),
      title=dict(
          text=(
              "3D Flux Space Trajectory: Philippine Seasonal Irradiance Dynamics"
          ),
          font=dict(color="#f8fafc", size=16),
      ),
  )

  return fig
