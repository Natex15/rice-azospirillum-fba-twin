# TEST_INFRA.md: Automated End-to-End Test Suite Architecture & Methodology

## 1. Executive Summary & Test Philosophy

This document defines the architectural blueprint, mathematical assertions, decoupled fixture harness, and progressive execution methodology for the automated End-to-End (E2E) test suite of the **Rice (*Oryza sativa indica*) – *Azospirillum brasilense* Two-Compartment Flux Balance Analysis (FBA) Simulation & 3D WebGL Digital Twin** platform.

### 1.1 Opaque-Box Specification-Driven Paradigm
The test suite operates strictly under an **opaque-box requirement-driven model**:
- Tests are synthesized directly from the system specifications in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and the authoritative architectural discovery blueprints (`infra_architecture_spec.md`, `tier1_tier2_spec.md`, `tier3_tier4_spec.md`).
- Tests interact solely with public module APIs, classes, and typed dataclass contracts (`FBASolution`, `PhasePlane2DResult`, `AmmoniumBifurcationResult`, `HypothesisTestResult`, `Sweep1DResult`, `Experiment1Result`).
- Tests are invariant to internal code restructuring, private variable renaming, or refactoring.

### 1.2 Zero External Binary Solver Dependency
A core tenet of this test infrastructure is deterministic standalone portability:
- All metabolic linear programming optimizations execute using `scipy.optimize.linprog(method='highs')`.
- This eliminates any mandatory runtime dependence on external compiled C/C++ solver binaries (e.g. GLPK, CPLEX, Gurobi).
- When COBRApy is present, optional numerical parity tests verify agreement with SciPy HiGHS to within $|Z_{\text{cobra}} - Z_{\text{highs}}| \le 10^{-5}$.

### 1.3 Progressive Testability & Decoupled Fixture Harness
To allow independent validation across developmental milestones without premature blocking:
- The test harness implements a **decoupled progressive testability pattern**: self-contained reference implementations (`ReferenceModel`, `ReferenceSimulator`, `ReferenceVisualizer`, `ReferenceApp`) are built into the test framework to evaluate interface contracts and mathematical properties immediately.
- When production modules (`model_builder.py`, `simulation_engine.py`, `visualizer_3d.py`, `app.py`) become importable, the test suite dynamically hooks into production classes and functions, confirming end-to-end integration.

---

## 2. 4-Tier Test Suite Hierarchy

The test catalog spans four rigorous testing tiers structured across requirements R1 through R5:

```
==================================================================================================
                                    4-TIER E2E TEST CATALOG
==================================================================================================

  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 1: FEATURE COVERAGE (R1 - R4)                                                         │
  │ • R1 Metabolic Model: 4 compartments, 24 reactions, S-matrix rank, HiGHS solver, SBML/JSON │
  │ • R2 Simulation Engine: Single-point FBA, two-stage slack starvation solver, 1D/2D/3D grids│
  │ • R3 3D Visualizer: Three.js WebGL HTML, Plotly 3D vector scene, Pareto/Collapse surfaces │
  │ • R4 App Components: Sidebar limits, 5-tab layout, KPI calculators, CSV/Markdown export  │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 2: BOUNDARY & CORNER CASES                                                            │
  │ • BC1 Extreme Light: Complete darkness (0.0%), sub-basal light (5.0%), hyper-light (150.0%)│
  │ • BC2 ATP Maintenance: Exact 2.5 bound, starvation deficit quantification (2.5 - ATP_ach) │
  │ • BC3 Genotype Bounds: WT ([0,10]), OE (lb=2.0), Knockdown (ub=3.0), KO ([0,0])           │
  │ • BC4 Soil NH4+ Extremes: Pure diazotrophy (0.0), hyper-fertilization (100.0), parasitism  │
  │ • BC5 Symbiotic Decoupling: Axenic uninoculated control (inoculated=False)                 │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 3: COMBINATORIAL & CROSS-FEATURE INTERACTIONS                                         │
  │ • 96-State Factorial Matrix: 4 Light x 4 Genotypes x 3 Soil NH4+ x 2 Inoculation States   │
  │ • Solver Fallback Stress: 30% typhoon light + OsSWEET11_OE (lb=2.0) lethal starvation cliff│
  │ • 2D/3D Tensor Shape Integrity: Coordinate alignment, slice parity with 1D sweeps          │
  │ • Visualization Data Contracts: Plotly go.Figure trace schemas and Three.js HTML tokens   │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 4: REAL-WORLD APPLICATION SCENARIOS & SCIENTIFIC HYPOTHESES (R5)                      │
  │ • Experiment 1 (RQ1): Philippine seasonal insolation monotonicity & carbon plasticity      │
  │ • Experiment 2 (RQ2): Formal H0 vs H1 hypothesis testing at 50% light -> [LB*, UB*] window │
  │ • Experiment 3 (RQ3): 3D soil ammonium collapse landscape & [NH4+]crit bisection           │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
==================================================================================================
```

---

## 3. Progressive Test Execution Layout & Pytest Markers

The test suite incorporates custom Pytest markers allowing fine-grained execution by feature milestone or testing tier:

### 3.1 Pytest Marker Taxonomy
- `r1_metabolic_model`: Tests validating the stoichiometric two-compartment model in `model_builder.py`.
- `r2_simulation_engine`: Tests validating the FBA simulation engine and tensor generators in `simulation_engine.py`.
- `r3_visualizer_3d`: Tests validating Engine A (Three.js WebGL & Plotly 3D vector twin) and Engine B (3D phase surfaces) in `visualizer_3d.py`.
- `r4_app_components`: Tests validating UI logic, sidebar parameters, KPI calculators, and export pipelines in `app.py`.
- `r5_scientific_hypotheses`: Tests validating scientific empirical acceptance criteria (RQ1, RQ2, RQ3).
- `tier1`: All Tier 1 functional feature coverage test cases.
- `tier2`: All Tier 2 boundary and corner case test cases.
- `tier3`: All Tier 3 multi-feature combinatorial and solver fallback interaction test cases.
- `tier4`: All Tier 4 real-world agro-climatic scenarios and hypothesis tests.

### 3.2 Command-Line Execution Matrix

```powershell
# 1. Run Complete E2E Test Suite (All Tiers 1-4, R1-R5)
pytest tests/test_simulation.py -v --tb=short

# 2. Run Milestone-Specific Tests:
# Milestone 1: Metabolic Model
pytest tests/test_simulation.py -m r1_metabolic_model -v

# Milestone 2: Simulation Engine
pytest tests/test_simulation.py -m r2_simulation_engine -v

# Milestone 3: 3D Visualization Suite
pytest tests/test_simulation.py -m r3_visualizer_3d -v

# Milestone 4: Streamlit UI Components & Data Pipelines
pytest tests/test_simulation.py -m r4_app_components -v

# Milestone 5 / Scientific Hypotheses Acceptance
pytest tests/test_simulation.py -m r5_scientific_hypotheses -v

# 3. Run Tier-Specific Tests:
pytest tests/test_simulation.py -m tier1 -v
pytest tests/test_simulation.py -m tier2 -v
pytest tests/test_simulation.py -m tier3 -v
pytest tests/test_simulation.py -m tier4 -v
```

---

## 4. Mathematical & Biological Invariant Assertions

The test infrastructure enforces deterministic invariant checks on every optimization output:

### 4.1 Steady-State Stoichiometric Mass Balance
Every feasible solution vector $v \in \mathbb{R}^{24}$ must satisfy Kirchhoff-like conservation for all internal metabolites:
$$\|S \cdot v\|_\infty = \max_{i} \left| \sum_{j=1}^{24} S_{ij} v_j \right| < 10^{-6}$$

### 4.2 Thermodynamic Flux Bound Compliance
All reaction fluxes must remain strictly within their lower and upper bounds:
$$lb_j - 10^{-6} \le v_j \le ub_j + 10^{-6} \quad \forall j \in \{1, \dots, 24\}$$

### 4.3 Mandatory Non-Growth Root Maintenance ATP
In any metabolically viable state, root cellular maintenance is strictly non-negotiable:
$$v_{\text{ATPM\_r}} \ge 2.5\,\text{mmol gDW}^{-1}\text{ h}^{-1}$$
When light is insufficient or excessive sucrose exudation deprives the root:
- The two-stage slack relaxation algorithm relaxes $lb_{\text{ATPM\_r}} = 0.0$ and solves $\max v_{\text{ATPM\_r}}$.
- The systemic starvation deficit is quantified as:
  $$\Delta \text{ATP} = \max(0.0, 2.5 - v_{\text{ATPM\_r}})$$
- The solution state is flagged: $is\_starved = \text{True}$, $status = \text{"starvation"}$, and vegetative host growth collapses to $BIOMASS_{\text{rice}} = 0.0$.

### 4.4 Net Host Benefit Index (NHBI) Formulation
$$\text{NHBI}([\text{NH}_4^+]_{\text{soil}}, v_{\text{SWEET11}}) = BIOMASS_{\text{rice}}(\text{inoculated}) - BIOMASS_{\text{rice}}(\text{uninoculated})$$
- Mutualistic state: $\text{NHBI} > 0 \land v_{\text{NITROGENASE\_ab}} > 0$.
- Parasitic state: $\text{NHBI} < 0 \land v_{\text{NITROGENASE\_ab}} = 0 \land v_{\text{SUC\_uptake\_ab}} > 0$.
- Bifurcation tipping plane: $\text{NHBI} = 0.0$.

---

## 5. Mock Fixture Designs for Decoupled Testing

The test harness provides pre-configured, mathematically consistent session- and module-scoped fixtures:

| Fixture Name | Type | Scope | Description |
|:---|:---:|:---:|:---|
| `expected_metabolite_ids` | `list[str]` | `session` | Authoritative list of 21 internal/boundary metabolite IDs. |
| `expected_reaction_ids` | `list[str]` | `session` | Authoritative list of 24 metabolic and exchange reaction IDs. |
| `default_model` | Model Object | `module` | 4-compartment model instance with WT parameters and 100% light. |
| `mock_fba_solution` | `FBASolution` | `session` | Feasible optimal solution ($BIOMASS_{\text{rice}} = 0.125, v_{\text{ATPM\_r}} = 2.5$). |
| `mock_starved_solution` | `FBASolution` | `session` | Starved solution ($is\_starved = \text{True}, \Delta \text{ATP} = 2.5, BIOMASS = 0.0$). |
| `mock_phase_plane_result`| `PhasePlane2DResult` | `session` | Synthetic $(20 \times 25)$ tensor field with starvation mask & Pareto ridge. |
| `mock_bifurcation_result`| `AmmoniumBifurcationResult` | `session` | Synthetic $(20 \times 20)$ bifurcation grid with $[\text{NH}_4^+]_{\text{crit}} \approx 3.45$. |
| `mock_seasonal_result` | `Experiment1Result` | `session` | Discrete solutions for the 4 Philippine seasons and continuous trajectory. |

---

## 6. Full Coverage Thresholds & Target Metrics

| Testing Dimension | Metric Target | Verification Method |
|:---|:---:|:---|
| **Mathematical Invariance** | $100\%$ compliance | Assert $\max|S \cdot v| < 10^{-6}$ and $lb \le v \le ub$ across all tests |
| **Solver Fallback** | $100\%$ zero binary | Verify HiGHS solves LPs without external binaries |
| **Feature Coverage** | $100\%$ (R1 to R5) | Explicit test cases for every requirement and feature catalog entry |
| **Edge & Corner Cases** | $100\%$ boundary checks | Darkness ($0\%$), hyper-light ($150\%$), $lb=2.5$ enforcement, 4 genotypes |
| **Combinatorial Matrix** | $96$ Factorial States | $4 \times 4 \times 3 \times 2$ grid parameterization without crashes |
| **Scientific Hypotheses** | $3$ Empirical Questions | Strict RQ1 monotonicity, RQ2 $H_1$ window $[LB^*, UB^*]$, RQ3 $[\text{NH}_4^+]_{\text{crit}}$ root |
| **Total Automated Tests** | $137$ Executable Tests | Complete coverage across Tier 1 (29), Tier 2 (BC1-BC5), Tier 3 (99), Tier 4 (8) |
| **Test Suite Pass Rate** | $100\%$ Pass (0 Failures, 0 Warnings) | `pytest tests/test_simulation.py -v` exits with code 0 |

---

## 7. Remediation & Invariant Rigor Verification (Iteration 2)

The test suite incorporates complete automated invariant and contract validations:
1. **Diazotrophic Stoichiometry Verification**: `test_nitrogenase_stoichiometry` enforces $1\,\text{N}_2[\text{rhizo}] + 16\,\text{ATP}[\text{a\_bras}] + 8\,\text{Fd}_{red}[\text{a\_bras}] \rightarrow 2\,\text{NH}_4^+[\text{a\_bras}] + 1\,\text{H}_2[\text{rhizo}]$ with backward-compatible conditional ferredoxin validation.
2. **Hyper-Light Boundary**: `test_hyper_light_boundary` validates feasibility, positive vegetative biomass ($BIOMASS_{\text{rice}} > 0$), maintenance satisfaction ($v_{\text{ATPM\_r}} \ge 2.5$), and Kirchhoff mass conservation at $150.0\%$ solar radiation.
3. **96-State Grid Invariant Verification**: `test_t3_comb_001_factorial_grid` enforces steady-state stoichiometric mass conservation ($\max\|S \cdot v\| < 10^{-5}$) and thermodynamic active flux bounds compliance ($lb_j \le v_j \le ub_j$) across all viable states.
4. **Public API Contract**: `test_model_optimize_method` directly verifies `default_model.optimize()` returning valid optimal solutions without external wrappers.
5. **Root Maintenance Enforcement**: `assert_maintenance_satisfied(sol)` is actively integrated into optimal, starvation, and hyper-light test paths.
6. **Zero-Warning Pytest Environment**: Root `pytest.ini` pre-registers all 9 custom markers, ensuring warning-free execution.
