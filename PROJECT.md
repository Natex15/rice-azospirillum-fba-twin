# Project: Rice-Azospirillum Two-Compartment FBA & 3D WebGL Digital Twin

## Architecture
A multi-compartment Flux Balance Analysis (FBA) simulation and interactive 3D WebGL / Plotly 3D + Streamlit scientific application coupling an Indica rice metabolic network (*Oryza sativa indica*) with a rhizosphere diazotrophic bacterium (*Azospirillum brasilense*) under simulated Philippine seasonal light conditions.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Streamlit Web Application               │
                  │                        (app.py)                         │
                  └──────┬───────────────────────┬────────────────────┬─────┘
                         │                       │                    │
                         ▼                       ▼                    ▼
             ┌───────────────────────┐   ┌──────────────┐   ┌───────────────────┐
             │  3D Visualizer Suite  │   │  Simulation  │   │   Metabolic Model │
             │   (visualizer_3d.py)  │   │    Engine    │   │ (model_builder.py)│
             │ - Three.js WebGL      │◄──┤(simulation_  │◄──┤ - 4 Compartments  │
             │ - Plotly 3D Vectors   │   │  engine.py)  │   │ - 24 Reactions    │
             │ - 3D Phase Surfaces   │   │ - FBA Solvers│   │ - HiGHS Fallback  │
             └───────────────────────┘   │ - 3D Tensors │   │ - SBML/JSON I/O   │
                                         └──────────────┘   └───────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | ENV_DEPS | Python dependencies specification (`requirements.txt`) | M1 | Survey |
| 2 | MODEL_COMPARTMENTS | 4-compartment topology (`[r_shoot]`, `[r_root]`, `[rhizosphere]`, `[a_bras]`) | M1 | Survey |
| 3 | MODEL_REACTIONS | 16 core reactions + 8 exchange/sink reactions with stoichiometric mass balance | M1 | Survey |
| 4 | MODEL_MAINTENANCE_STARVATION | Non-growth cellular maintenance `ATPM_r` ($lb = 2.5$) and carbon starvation detection | M1 | Survey |
| 5 | MODEL_NITROGENASE | Diazotrophic `NITROGENASE_ab` stoichiometry ($16\text{ ATP} + 8\text{ Fd\_red} \rightarrow 2\text{ NH}_4^+ + \text{H}_2$) | M1 | Survey |
| 6 | MODEL_GENOTYPES | `SWEET11_export` parameterization for WT, OE ($lb > 0$), Knockdown ($ub = 30\% \text{WT}$), KO ($lb=0, ub=0$) | M1 | Survey |
| 7 | MODEL_SOLVER_FALLBACK | Automatic `scipy.optimize.linprog(method='highs')` fallback with zero solver binaries | M1 | Survey |
| 8 | MODEL_SBML_JSON | SBML `.xml` and `.json` model importer/exporter | M1 | Survey |
| 9 | SIM_SINGLE_POINT | Single-point FBA solver with two-stage slack carbon starvation detection | M2 | Survey |
| 10 | SIM_EXP1_SEASONAL | Experiment 1 Philippine seasonal light regimes (100% Dry, 70% Monsoon, 50% Rainy, 30% Typhoon) | M2 | Survey |
| 11 | SIM_EXP2_PHASE_PLANE | Experiment 2 OsSWEET11 3D phase plane tensor grid ($X, Y, Z_1, Z_2, Starvation\_Mask$) | M2 | Survey |
| 12 | SIM_EXP2_HYPOTHESIS | Experiment 2 formal $H_0$ vs $H_1$ hypothesis testing algorithm at 50% light returning $[LB, UB]$ window | M2 | Survey |
| 13 | SIM_EXP3_AMMONIUM_COLLAPSE | Experiment 3 3D soil ammonium collapse tensor grid & Net Host Benefit Index ($NHBI$) | M2 | Survey |
| 14 | SIM_EXP3_BIFURCATION | Experiment 3 $[\text{NH}_4^+]_{\text{crit}}$ bifurcation root-finding (mutualism $\rightarrow$ parasitism switch) | M2 | Survey |
| 15 | VIZ_THREEJS_WEBGL | Engine A interactive Three.js WebGL canvas embedded in Streamlit with live particle flows & health colors | M3 | Survey |
| 16 | VIZ_PLOTLY_3D_ANATOMICAL | Engine A interactive Plotly 3D vector anatomical scene (`go.Mesh3d` + `go.Scatter3d` + `go.Cone`) | M3 | Survey |
| 17 | VIZ_PLOTLY_3D_PARETO | Engine B `plot_3d_pareto_surface()` (Light $\times$ SWEET11 $\times$ Biomass with nitrogenase color & cliff) | M3 | Survey |
| 18 | VIZ_PLOTLY_3D_COLLAPSE | Engine B `plot_3d_ammonium_collapse_surface()` with $Z=0$ threshold plane | M3 | Survey |
| 19 | VIZ_PLOTLY_3D_TRAJECTORY | Engine B `plot_3d_flux_space_trajectory()` across 4 seasonal light regimes | M3 | Survey |
| 20 | APP_SIDEBAR | Sidebar controls (seasonal presets, light slider, OsSWEET11 presets/sliders, NH4+ slider, camera, SBML uploader) | M4 | Survey |
| 21 | APP_TAB1_DIGITAL_TWIN | Tab 1 3D Live Rhizosphere Digital Twin & Metabolic Dashboard (Three.js & Plotly 3D side-by-side + KPIs) | M4 | Survey |
| 22 | APP_TAB2_SEASONAL | Tab 2 Philippine Seasonal Light Regimes (3D trajectory & 2D comparisons) | M4 | Survey |
| 23 | APP_TAB3_PHASE_PLANE | Tab 3 $H_0$ vs $H_1$ Hypothesis Evaluator & 3D Phase Plane | M4 | Survey |
| 24 | APP_TAB4_COLLAPSE | Tab 4 3D Soil Ammonium Collapse Surface & $[\text{NH}_4^+]_{\text{crit}}$ Bifurcation | M4 | Survey |
| 25 | APP_TAB5_RAW_EXPORT | Tab 5 Raw Flux Tables, Sensitivity Analysis & CSV/Markdown Manuscript Exporter | M4 | Survey |
| 26 | TEST_E2E_INFRA | Requirement-driven opaque-box E2E test suite across Tiers 1-4 | M_E2E | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Metabolic Model | `requirements.txt`, `model_builder.py` (Features 1-8) | none | DONE |
| M2 | Simulation Engine | `simulation_engine.py` (Features 9-14) | M1 contracts | DONE |
| M3 | 3D Visualizer Suite | `visualizer_3d.py` (Features 15-19) | M2 contracts | DONE |
| M4 | Streamlit Application | `app.py`, `README.md` (Features 20-25) | M1, M2, M3 | DONE |
| M_E2E | E2E Testing Track | `TEST_INFRA.md`, `tests/test_simulation.py`, `TEST_READY.md` (Feature 26) | none (opaque-box) | DONE |
| M_FINAL | Final Milestone | 100% E2E test pass (Tiers 1-4) + Adversarial Coverage Hardening (Tier 5) | M4, M_E2E | IN_PROGRESS |

## Interface Contracts

### model_builder.py ↔ simulation_engine.py
```python
class TwoCompartmentMetabolicModel:
    """Encapsulates reactions, metabolites, S matrix, bounds, and solver interfaces."""
    def get_reaction_bounds(self, reaction_id: str) -> tuple[float, float]: ...
    def set_reaction_bounds(self, reaction_id: str, lb: float, ub: float) -> None: ...
    def set_genotype(self, genotype: str, lb_custom: float | None = None, ub_custom: float | None = None) -> None: ...
    def optimize(self, objective: str = "BIOMASS_rice", maximize: bool = True) -> "FBASolution": ...
    def to_sbml(self, filepath: str) -> None: ...
    def to_json(self, filepath: str) -> None: ...

def build_two_compartment_model() -> TwoCompartmentMetabolicModel: ...
def load_model_from_file(filepath_or_buffer) -> TwoCompartmentMetabolicModel: ...
```

### simulation_engine.py ↔ visualizer_3d.py
```python
@dataclass
class FBASolution:
    status: str            # "optimal", "infeasible", "starved"
    objective_value: float
    fluxes: dict[str, float]
    is_starved: bool
    atpm_deficit: float

@dataclass
class PhasePlane2DResult:
    X_light: np.ndarray          # 2D grid of Light % (Ny, Nx)
    Y_sweet11: np.ndarray        # 2D grid of SWEET11 export (Ny, Nx)
    Z_rice_biomass: np.ndarray   # 2D grid of Rice Biomass (Ny, Nx)
    Z_nitrogenase: np.ndarray    # 2D grid of Nitrogenase flux (Ny, Nx)
    Starvation_Mask: np.ndarray  # 2D boolean mask where ATPM_r fails (Ny, Nx)

@dataclass
class AmmoniumBifurcationResult:
    X_nh4: np.ndarray            # 2D grid of soil NH4+ (Ny, Nx)
    Y_sweet11: np.ndarray        # 2D grid of SWEET11 export (Ny, Nx)
    Z_nhbi: np.ndarray           # 2D grid of Net Host Benefit Index (Ny, Nx)
    Z_nitrogenase: np.ndarray    # 2D grid of Nitrogenase flux (Ny, Nx)
    critical_nh4_threshold: float # Exact [NH4+]crit transition point
```

### visualizer_3d.py ↔ app.py
```python
# Engine A:
def render_threejs_rhizosphere(light_pct: float, sweet11_flux: float, nh4_flux: float, is_starved: bool, is_parasitic: bool, height: int = 650) -> str:
    """Returns self-contained HTML/JS string for st.components.v1.html."""
    ...

def plot_3d_anatomical_scene(fluxes: dict[str, float], is_starved: bool, is_parasitic: bool) -> go.Figure:
    """Returns interactive Plotly 3D vector scene with Mesh3d, Scatter3d, and Cones."""
    ...

# Engine B:
def plot_3d_pareto_surface(phase_result: PhasePlane2DResult, current_genotype: str | None = None) -> go.Figure: ...
def plot_3d_ammonium_collapse_surface(bifurcation_result: AmmoniumBifurcationResult) -> go.Figure: ...
def plot_3d_flux_space_trajectory(exp1_result: "Experiment1Result") -> go.Figure: ...
```

## Code Layout
```
c:\Users\Admin\Documents\Rice Simulation/
├── requirements.txt            # Python dependencies (numpy, scipy, pandas, plotly, streamlit, cobra, pytest)
├── model_builder.py            # R1: 4-compartment FBA model, SciPy HiGHS fallback, SBML/JSON parser
├── simulation_engine.py        # R2: FBA solvers, 1D/2D/3D tensor generators, RQ1/RQ2/RQ3 experiments
├── visualizer_3d.py            # R3: Engine A (Three.js WebGL + Plotly 3D) & Engine B (3D phase surfaces)
├── app.py                      # R4: 5-Tab Streamlit scientific web application
├── README.md                   # User manual, scientific background, and quickstart
├── tests/
│   └── test_simulation.py      # Automated unit & E2E test suite
├── TEST_INFRA.md               # E2E test suite architecture & methodology
└── TEST_READY.md               # E2E test suite completion signal & coverage matrix
```
