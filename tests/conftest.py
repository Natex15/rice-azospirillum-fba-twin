"""Pytest configuration and global hooks for Rice Simulation test suite."""


def pytest_configure(config):
  """Register custom markers to prevent PytestUnknownMarkWarning."""
  for mark, desc in [
      ("r1_metabolic_model", "Tests for Requirement 1"),
      ("r2_simulation_engine", "Tests for Requirement 2"),
      ("r3_visualizer_3d", "Tests for Requirement 3"),
      ("r4_app_components", "Tests for Requirement 4"),
      ("r5_scientific_hypotheses", "Tests for Requirement 5"),
      ("tier1", "Tier 1 feature coverage"),
      ("tier2", "Tier 2 boundary cases"),
      ("tier3", "Tier 3 combinatorial tests"),
      ("tier4", "Tier 4 scientific scenarios"),
  ]:
    config.addinivalue_line("markers", f"{mark}: {desc}")
