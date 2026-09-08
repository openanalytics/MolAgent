"""Build the feature-generator dict for a training run.

Extends the default offline generators (Bottleneck variants, rdkit, fps_*) with
any molfeat-based generators that were explicitly requested but are not in the
default set.  Uses automol's own wrapper classes so the generators implement
the full FeatureGenerator interface (get_nb_features, generate, …).

Supported extra keys
--------------------
Offline (needs: molfeat):
  desc2D, ecfp-count, fcfp-count, maccs, avalon, secfp, estate, erg
  → MolfeatFPVecTransformer

HuggingFace pretrained (needs: molfeat + transformers + tokenizers):
  ChemGPT-4.7M/19M/1.2B  → MolfeatPretrainedHFTransformer(notation='selfies')
  ChemBERTa-77M-MTR/MLM, MolT5  → MolfeatPretrainedHFTransformer(notation='smiles')

DGL pretrained (needs: molfeat + dgl + dgllife):
  gin_supervised_edgepred/infomax/contextpred/masking, jtvae_zinc_no_kl
  → MolfeatPretrainedDGLTransformer

Graphormer pretrained (needs: molfeat + graphormer):
  pcqm4mv2_graphormer_base, pcqm4mv1_graphormer_base, …
  → MolfeatGraphormerTransformer
"""
from __future__ import annotations

# Keys that map to MolfeatFPVecTransformer(kind=key)
_MOLFEAT_FP = frozenset({
    "desc2D", "ecfp-count", "fcfp-count", "maccs",
    "avalon", "secfp", "estate", "erg",
})

# Keys that map to MolfeatPretrainedHFTransformer — notation varies by model family
# ChemGPT models use SELFIES notation; ChemBERTa and MolT5 use SMILES
_MOLFEAT_HF_SELFIES = frozenset({
    "ChemGPT-4.7M", "ChemGPT-19M", "ChemGPT-1.2B",
})
_MOLFEAT_HF_SMILES = frozenset({
    "ChemBERTa-77M-MTR", "ChemBERTa-77M-MLM", "MolT5",
})

# Keys that map to MolfeatPretrainedDGLTransformer(kind=key) — needs dgl + dgllife
_MOLFEAT_DGL = frozenset({
    "gin_supervised_edgepred", "gin_supervised_infomax",
    "gin_supervised_contextpred", "gin_supervised_masking",
    "jtvae_zinc_no_kl",
})

# Keys that map to MolfeatGraphormerTransformer(kind=key) — needs graphormer
_MOLFEAT_GRAPHORMER = frozenset({
    "pcqm4mv2_graphormer_base", "pcqm4mv1_graphormer_base",
    "pcqm4mv1_graphormer_base_for_molhiv", "graphormer_pretrained",
})


def build_feature_generators(
    feature_keys: list[str],
    radius: int = 2,
    nbits: int = 2048,
) -> dict:
    """Return the feature-generator dict for the requested keys.

    Always includes the full default offline set. Extra molfeat generators are
    added only when explicitly requested.
    """
    from automol.feature_generators import (
        retrieve_default_offline_generators,
        MolfeatFPVecTransformer,
        MolfeatPretrainedHFTransformer,
        MolfeatPretrainedDGLTransformer,
        MolfeatGraphormerTransformer,
    )

    generators = retrieve_default_offline_generators(radius=radius, nbits=nbits)

    for key in feature_keys:
        if key in generators:
            continue
        if key in _MOLFEAT_FP:
            generators[key] = MolfeatFPVecTransformer(kind=key)
        elif key in _MOLFEAT_HF_SELFIES:
            generators[key] = MolfeatPretrainedHFTransformer(kind=key, notation="selfies")
        elif key in _MOLFEAT_HF_SMILES:
            generators[key] = MolfeatPretrainedHFTransformer(kind=key, notation="smiles")
        elif key in _MOLFEAT_DGL:
            generators[key] = MolfeatPretrainedDGLTransformer(kind=key)
        elif key in _MOLFEAT_GRAPHORMER:
            generators[key] = MolfeatGraphormerTransformer(kind=key)

    return generators
