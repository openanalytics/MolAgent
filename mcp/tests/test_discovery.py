"""Tests for mcp/_discovery.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prolif_and_affgraph_flagged_as_requiring_3d_structure():
    from _discovery import list_feature_generators

    generators = list_feature_generators()
    assert "prolif" in generators
    assert "AffGraph" in generators
    assert isinstance(generators["prolif"], dict)
    assert generators["prolif"]["requires_3d_structure"] is True
    assert isinstance(generators["AffGraph"], dict)
    assert generators["AffGraph"]["requires_3d_structure"] is True


def test_existing_generators_are_unaffected():
    from _discovery import list_feature_generators

    generators = list_feature_generators()
    assert "_note" in generators
    assert isinstance(generators["_note"], str)
    # At least one pre-existing plain-string entry must survive unchanged.
    assert isinstance(generators.get("Bottleneck", generators.get("rdkit")), str)
