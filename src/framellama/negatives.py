"""Negative-candidate selection and filtering.

From the original research generator (lines 234-270 and 370-426). Two
filters are applied when the generator builds negative examples:

1. Frame-level filter (``negative_frames``). A candidate negative frame Fn is
   kept for a target frame Ft only if it is not directly related to Ft and
   sim(A(Ft), A(Fn)) < ``NEGATIVE_THRESHOLD``, where A(F) is the text bundle
   of F (definition, FEs, semantic types, LU definitions, relations).

2. Candidate-level filter (``get_first_low_similarity_candidate``). The first
   distractor whose similarity to the correct answer is below a threshold is
   used. The thresholds (0.5 or 0.4) are set at each call site in
   ``dataset_builder.py``, exactly as in the original code.

Similarities are cosine similarities of sentence-transformers embeddings.
"""

from . import framenet_reader

# Defaults are the values used for the ACL 2026 corpus.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEGATIVE_THRESHOLD = 0.50

model = None
util = None
embedding_cache = {}
bundle_similarity = None
frame_positions = {}
all_frames = []


def load_model(name=EMBEDDING_MODEL):
    """Load the sentence-embedding model (once)."""
    global model, util, EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as st_util
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for negative filtering: "
            "pip install sentence-transformers"
        ) from exc
    EMBEDDING_MODEL = name
    model = SentenceTransformer(name)
    util = st_util
    embedding_cache.clear()
    return model


def get_embedding(text):
    text = text.strip()
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True)
    return embedding_cache[text]


def get_first_low_similarity_candidate(correct_text,
                                       candidate_list,
                                       threshold=0.5):
    """
    Return first candidate with similarity < threshold (early break)
    """

    if not correct_text or not candidate_list:
        return None, None

    correct_emb = get_embedding(correct_text)

    for c in candidate_list:
        if not isinstance(c, str) or not c.strip():
            continue

        emb = get_embedding(c)
        score = util.cos_sim(correct_emb, emb).item()

        if score < threshold:
            return c, score   # break immediately

    return None, None  # if nothing found


def frame_bundle(frame):
    """A(F): frame definition, FEs, semantic types, LU definitions and relations."""
    translation_dict = framenet_reader.translation_dict
    norm = framenet_reader.norm
    frame_key = norm(frame.name.replace("_", " "))
    parts = [frame.name.replace("_", " "), translation_dict.get((frame_key, frame_key), "")]

    for fe_name, fe in frame.FE.items():
        fe_label = fe_name.replace("_", " ")
        parts.append(f"{fe_label}: {translation_dict.get((frame_key, norm(fe_label)), '')}")
        if fe.semType:
            parts.append(fe.semType.name.replace("_", " "))

    for lu_name, lu in frame.lexUnit.items():
        parts.append(f"{lu_name}: {lu.definition}")

    for rel in frame.frameRelations:
        parts.append(f"{rel['type']['name']}: {rel['superFrameName']} {rel['subFrameName']}")

    return " ".join(part.strip() for part in parts if part.strip())


def build_negative_index(frames, threshold=NEGATIVE_THRESHOLD, show_progress=True):
    """Embed the bundle of every frame and store the frame-by-frame similarity matrix."""
    global bundle_similarity, frame_positions, all_frames, NEGATIVE_THRESHOLD
    NEGATIVE_THRESHOLD = threshold
    all_frames = list(frames)
    bundle_embeddings = model.encode(
        [frame_bundle(f) for f in all_frames],
        convert_to_tensor=True,
        batch_size=64,
        show_progress_bar=show_progress
    )
    bundle_similarity = util.cos_sim(bundle_embeddings, bundle_embeddings)
    frame_positions = {f.name: i for i, f in enumerate(all_frames)}


def negative_frames(name):
    """Frames allowed as negatives for a target frame."""
    target = name.replace(" ", "_")
    excluded = framenet_reader.related_frames(target) | {target}
    similarities = bundle_similarity[frame_positions[target]].tolist()
    return [
        f for f, similarity in zip(all_frames, similarities)
        if similarity < NEGATIVE_THRESHOLD and f.name not in excluded
    ]
