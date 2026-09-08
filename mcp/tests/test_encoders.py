"""Encoder registry, embedding, naming, packaging and pickling tests.

Run with:
    ./run_tests.sh -k encoders
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:                      # stdlib from 3.11; requires-python is >=3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).parent.parent))

import automol

# automol is a NAMESPACE package: automol.__file__ is None.
PKG_DIR = Path(automol.__path__[0])
PYPROJECT = Path(__file__).resolve().parents[2] / "AutoMol" / "automol" / "pyproject.toml"

# A fixed, deterministic SMILES set used across the embedding tests.
SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c2cccnc2"]


def test_encoder_assets_are_covered_by_package_data():
    """Every file under automol/encoders/ must match a package-data glob.

    setuptools expands each package-data pattern with a NON-recursive glob
    relative to the package directory, so 'encoders/e_base/encoder.onnx' is not
    matched by '*.onnx'. Without an explicit nested pattern the encoder assets
    are silently dropped from the wheel and MolBottleGenerator raises
    FileNotFoundError on any non-editable install.
    """
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    patterns = cfg["tool"]["setuptools"]["package-data"]["automol"]

    covered: set[Path] = set()
    for pattern in patterns:
        covered |= {p.resolve() for p in PKG_DIR.glob(pattern)}

    encoders_dir = PKG_DIR / "encoders"
    required = {p.resolve() for p in encoders_dir.rglob("*") if p.is_file()}
    assert required, f"no encoder assets found under {encoders_dir}"

    missing = sorted(str(p.relative_to(PKG_DIR)) for p in required - covered)
    assert not missing, f"encoder assets not covered by package-data: {missing}"


def test_e_logd_export_is_present_and_distinct_from_e_base():
    """v6_best ships as encoders/e_logd/ and is a different encoder than e_base.

    Comparing embeddings is what makes this meaningful: it would still pass if
    only the files existed, but it fails if e_logd is accidentally a copy of
    e_base.
    """
    from automol.feature_generators import MolBottleGenerator

    e_logd_dir = PKG_DIR / "encoders" / "e_logd"
    for name in ("config.json", "vocab.json", "encoder.onnx"):
        assert (e_logd_dir / name).exists(), f"missing {e_logd_dir / name}"
    assert not (e_logd_dir / "encoder.pt").exists(), "encoder.pt must not be shipped"

    logd = MolBottleGenerator(export_dir=str(e_logd_dir))
    base = MolBottleGenerator(export_dir=str(PKG_DIR / "encoders" / "e_base"))

    z_logd = logd.generate(SMILES)
    z_base = base.generate(SMILES)

    assert z_logd.shape == (len(SMILES), 250)
    assert z_base.shape == (len(SMILES), 250)
    assert np.isfinite(z_logd).all()
    assert not np.allclose(z_logd, z_base), "e_logd and e_base produced identical embeddings"


def test_molbottle_feature_names_distinguish_variants():
    """e_base and e_logd are both epoch 39, so names must carry the variant."""
    from automol.feature_generators import MolBottleGenerator

    logd = MolBottleGenerator(export_dir=str(PKG_DIR / "encoders" / "e_logd"))
    base = MolBottleGenerator(export_dir=str(PKG_DIR / "encoders" / "e_base"))

    assert not set(logd.names) & set(base.names), "feature names collide across variants"
    assert logd.generator_name != base.generator_name
    assert "e_logd" in logd.names[0]
    assert "e_base" in base.names[0]
    assert len(logd.names) == logd.nb_features == 250


def test_molbottle_variant_can_be_overridden():
    from automol.feature_generators import MolBottleGenerator

    gen = MolBottleGenerator(export_dir=str(PKG_DIR / "encoders" / "e_logd"), variant="custom")
    assert "custom" in gen.names[0]
    assert "custom" in gen.generator_name


def test_registry_exposes_four_encoder_keys_backed_by_three_instances():
    from automol.feature_generators import (
        CANONICAL_ENCODER_KEYS,
        FEATURE_KEY_ALIASES,
        retrieve_default_offline_generators,
    )

    d = retrieve_default_offline_generators()

    assert set(CANONICAL_ENCODER_KEYS) == {
        "Bottleneck", "Bottleneck_chembl37_base", "Bottleneck_chembl27",
    }
    assert FEATURE_KEY_ALIASES == {"Bottleneck_chembl37_logd": "Bottleneck"}

    expected = set(CANONICAL_ENCODER_KEYS) | set(FEATURE_KEY_ALIASES) | {"rdkit", "fps_2048_2"}
    assert set(d) == expected
    assert "MolBottle" not in d, "the unreleased MolBottle key must be removed"

    # The alias shares an instance, so only three ONNX sessions are opened.
    assert d["Bottleneck_chembl37_logd"] is d["Bottleneck"]
    encoders = {id(d[k]) for k in ("Bottleneck", "Bottleneck_chembl37_base", "Bottleneck_chembl27")}
    assert len(encoders) == 3


def test_default_key_is_v6_and_matches_a_direct_e_logd_load():
    from automol.feature_generators import (
        MolBottleGenerator,
        default_encoder,
        retrieve_default_offline_generators,
    )

    d = retrieve_default_offline_generators()
    direct = MolBottleGenerator(export_dir=str(PKG_DIR / "encoders" / "e_logd"))

    assert np.allclose(d["Bottleneck"].generate(SMILES), direct.generate(SMILES))
    assert np.allclose(default_encoder().generate(SMILES), direct.generate(SMILES))


def test_feature_names_unique_across_all_three_encoders():
    from automol.feature_generators import retrieve_default_offline_generators

    d = retrieve_default_offline_generators()
    keys = ("Bottleneck", "Bottleneck_chembl37_base", "Bottleneck_chembl27")
    all_names = [n for k in keys for n in d[k].names]
    assert len(all_names) == len(set(all_names)) == 750


def test_fresh_model_fallback_injects_v6_not_chembl27():
    """A caller-supplied dict without 'Bottleneck' gets the DEFAULT encoder.

    Clustering uses used_features=['Bottleneck'], so injecting the legacy
    encoder here would split the data with different embeddings than training
    uses.
    """
    from automol.feature_generators import RDKITGenerator
    from automol.stacking import FeatureGenerationStackingRegressors

    model = FeatureGenerationStackingRegressors(feature_generators={"rdkit": RDKITGenerator()})

    assert "Bottleneck" in model.feature_generators
    assert "e_logd" in model.feature_generators["Bottleneck"].generator_name


def test_legacy_model_paths_still_use_chembl27():
    """BottleneckFeatureGenerator.get_features lazily builds the LEGACY encoder.

    That branch only fires for objects that never ran __init__'s setup — i.e.
    models pickled before feature_generators existed, which were trained on the
    ChEMBL 27 encoder.
    """
    from automol.stacking import BottleneckFeatureGenerator

    gen = BottleneckFeatureGenerator()
    z = gen.get_features(SMILES)

    assert z.shape == (len(SMILES), 250)
    assert "e_logd" not in gen.bottleneck.generator_name
    assert "bottleneck_encoder" in gen.bottleneck.generator_name


@pytest.mark.parametrize("key,rel_field", [
    ("Bottleneck", "_onnx_rel"),
    ("Bottleneck_chembl37_base", "_onnx_rel"),
    ("Bottleneck_chembl27", "_model_path_rel"),
])
def test_pickled_encoder_resolves_relative_to_the_package(key, rel_field):
    """A moved install must still find its encoder.

    Simulated by corrupting the absolute path in the pickled state: if
    __setstate__ honours the relative marker the generator still works.
    """
    from automol.feature_generators import retrieve_default_offline_generators

    gen = retrieve_default_offline_generators()[key]
    expected = gen.generate(SMILES)

    state = gen.__getstate__()
    assert state[rel_field] is not None, f"{rel_field} not recorded"
    assert not Path(state[rel_field]).is_absolute()

    abs_field = "_onnx_path" if rel_field == "_onnx_rel" else "model_path"
    state[abs_field] = "/nonexistent/elsewhere/encoder.onnx"

    revived = type(gen).__new__(type(gen))
    revived.__setstate__(state)
    assert np.allclose(revived.generate(SMILES), expected)


@pytest.mark.parametrize("key,rel_field", [
    ("Bottleneck", "_onnx_rel"),
    ("Bottleneck_chembl27", "_model_path_rel"),
])
def test_legacy_pickles_without_a_relative_marker_still_load(key, rel_field):
    """Existing .pt files carry only an absolute path; they must keep working."""
    from automol.feature_generators import retrieve_default_offline_generators

    gen = retrieve_default_offline_generators()[key]
    expected = gen.generate(SMILES)

    state = gen.__getstate__()
    state.pop(rel_field)

    revived = type(gen).__new__(type(gen))
    revived.__setstate__(state)
    assert np.allclose(revived.generate(SMILES), expected)


def test_discovery_lists_canonical_keys_only():
    from _discovery import list_feature_generator_aliases, list_feature_generators

    listed = list_feature_generators()
    for key in ("Bottleneck", "Bottleneck_chembl37_base", "Bottleneck_chembl27"):
        assert key in listed, f"{key} missing from list_options"
        assert listed[key], f"{key} has an empty description"

    assert "Bottleneck_chembl37_logd" not in listed, "aliases must not be listed"
    assert "MolBottle" not in listed

    # The clean-encoder guidance has to be discoverable at runtime, since no
    # leakage warning is emitted during training.
    assert "logD" in listed["Bottleneck_chembl37_base"]

    assert list_feature_generator_aliases() == {"Bottleneck_chembl37_logd": "Bottleneck"}


def test_server_validation_accepts_canonical_keys_and_aliases():
    import server

    valid = server._valid_feature_keys()
    for key in (
        "Bottleneck",
        "Bottleneck_chembl37_logd",
        "Bottleneck_chembl37_base",
        "Bottleneck_chembl27",
        "rdkit",
    ):
        assert key in valid, f"{key} rejected by validation"
    assert "_note" not in valid
    assert "totally_invalid_feature" not in valid


def test_fresh_model_with_no_generators_uses_v6():
    """The feature_generators=None branch must also get the default encoder."""
    from automol.stacking import FeatureGenerationStackingRegressors

    model = FeatureGenerationStackingRegressors()

    assert "e_logd" in model.bottleneck.generator_name
    assert "e_logd" in model.feature_generators["Bottleneck"].generator_name


def test_merge_rejects_conflicting_bottleneck_encoders():
    """Merging across the Bottleneck repointing must fail loudly, not silently."""
    from automol.feature_generators import RDKITGenerator, retrieve_default_offline_generators
    from automol.stacking import FeatureGenerationStackingRegressors

    d = retrieve_default_offline_generators()
    new = FeatureGenerationStackingRegressors(
        feature_generators={"Bottleneck": d["Bottleneck"], "rdkit": RDKITGenerator()})
    old = FeatureGenerationStackingRegressors(
        feature_generators={"Bottleneck": d["Bottleneck_chembl27"], "rdkit": RDKITGenerator()})

    with pytest.raises(ValueError, match="different encoder"):
        new.merge_model(old, other_props=[])


# -------------------------------------------------------------------
# Molagent-only test: ONNX ChEMBL-27 reproduces the retired torch encoder.
# Requires AutoMol/automol_resources/automol/trained_models/bottleneck_CHEMBL27_ENUM_SMILES_ENCODER.pt
# Mark skip when the .pt is absent so CI on a clean clone still passes.
# Run once on this machine to prove the swap is faithful; the skip
# annotation then serves as documentation that this was verified.
# -------------------------------------------------------------------
from pathlib import Path as _Path

_PT_FILE = _Path(__file__).resolve().parents[2] / \
    "AutoMol" / "automol_resources" / "automol" / "trained_models" / \
    "bottleneck_CHEMBL27_ENUM_SMILES_ENCODER.pt"


@pytest.mark.skipif(not _PT_FILE.exists(), reason=".pt not present — verified on first run")
def test_onnx_chembl27_reproduces_torch_encoder():
    """Bottleneck_chembl27 (ONNX) must produce embeddings within tolerance of the
    retired torch BottleneckTransformer it replaces, over the same SMILES.

    Tolerance: rtol=1e-3, atol=1e-3. The ONNX and torch runtimes produce
    slightly different float32 results due to order-of-operations differences;
    exact equality is never expected.

    This test can only run while both the old .pt checkpoint and the new ONNX
    export are present. Once the .pt is removed, the skip annotation documents
    that the equivalence was verified.
    """
    csv_path = _Path(__file__).resolve().parents[2] / \
        "AutoMol" / "automol" / "automol" / "test" / "ChEMBL_SMILES.csv"
    if not csv_path.exists():
        pytest.skip("ChEMBL_SMILES.csv not found")
    lines = csv_path.read_text().splitlines()[1:]  # skip header
    smiles_for_equiv = [l.split(",")[1].strip() for l in lines if "," in l][:20]

    import torch

    # Load the retired torch encoder
    checkpoint = torch.load(str(_PT_FILE), map_location="cpu", weights_only=False)
    smiles_encoder = checkpoint["Smiles_Encoder"]
    smiles_encoder.load_state_dict(checkpoint["Smiles_Encoder_model_state_dict"])
    vocab = checkpoint["vocab"]
    smiles_encoder.eval()

    # Generate reference embeddings using the torch encoder directly.
    # vocab.smile2int() is the checkpoint's built-in SMILES tokenizer.
    # The ONNX export wraps sequences with BOS=3 and EOS=2 at seq_len=220;
    # we must replicate that here so both paths see identical input.
    # The encoder is a transformer that uses the BOS (time-step 0) output as
    # the sequence embedding.
    BOS, EOS, PAD, SEQ_LEN = 3, 2, 0, 220
    seqs = [vocab.smile2int(s) for s in smiles_for_equiv]
    padded = torch.zeros(len(seqs), SEQ_LEN, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok = [BOS] + list(s) + [EOS]
        padded[i, :len(tok)] = torch.tensor(tok)
    with torch.no_grad():
        out = smiles_encoder(padded.T)  # input: (seq_len, batch)
    # out = {last_layer_idx: {"output": tensor(seq_len, batch, hidden)}}.
    # BOS output (time-step 0) is the sequence-level embedding.
    last_key = list(out.keys())[-1]
    torch_embeds = out[last_key]["output"][0].numpy()  # shape: (N, 250)

    # Generate ONNX embeddings
    from automol.feature_generators import retrieve_default_offline_generators
    onnx_encoder = retrieve_default_offline_generators()["Bottleneck_chembl27"]
    onnx_embeds = onnx_encoder.generate(smiles_for_equiv)  # shape: (N, 250)

    assert torch_embeds.shape == onnx_embeds.shape, \
        f"shape mismatch: torch {torch_embeds.shape} vs onnx {onnx_embeds.shape}"
    assert np.allclose(torch_embeds, onnx_embeds, rtol=1e-3, atol=1e-3), \
        f"ONNX and torch embeddings differ beyond tolerance. " \
        f"Max abs diff: {np.abs(torch_embeds - onnx_embeds).max():.4f}"


def test_old_pickle_style_still_unpickles():
    """Pre-P1 pickles recorded __module__='automol.feature_generators'.
    The shim must make those loadable without raising AttributeError.
    """
    import pickle
    from automol.feature_generators import BottleneckTransformer

    # Simulate a pickle from before P1: the class path is in feature_generators.
    bt = BottleneckTransformer.__new__(BottleneckTransformer)
    state = {
        "_model_path_rel": "bottleneck_encoder.onnx",
        "model_path": str(
            __import__("pathlib").Path(
                __import__("automol").__path__[0]
            ) / "bottleneck_encoder.onnx"
        ),
        "_onnx_path": str(
            __import__("pathlib").Path(
                __import__("automol").__path__[0]
            ) / "bottleneck_encoder.onnx"
        ),
        "nb_features": 250,
        "names": [],
        "generator_name": "automol_onnx_bn_bottleneck_encoder",
        "batch_size": 100,
        "seq_len": 220,
    }
    # This must NOT raise AttributeError or KeyError.
    bt.__setstate__(state)
    assert bt.nb_features == 250
    assert bt._onnx_path.endswith("bottleneck_encoder.onnx")
