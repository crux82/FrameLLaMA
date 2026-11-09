"""FrameNet 1.7 access and lookup tables.

This module contains the FrameNet-reading part of the original research
generator (``framenet_dataset_generator.py``, lines 32-196, 273-330 and
354-386). The code is unchanged except that the lookup tables are built by
``load()`` instead of at import time, so that the FrameNet location can be
passed on the command line.

Two POS maps exist on purpose. The original script defined ``POS_MAP`` twice:
the first definition was used while the lookup tables were built, the second
(which overrides it) is used by every lookup at generation time. Both are kept
so the generated data is identical to the original.
"""

import os
import re

import pandas as pd

# Filled in by load()
fn = None          # FrameNet corpus reader
wn = None          # WordNet corpus reader
translation_dict = {}
example_dict = {}
lu_dict = {}
lu_def_dict = {}

# Files the NLTK FrameNet reader needs (same list NLTK uses for framenet_v17).
REQUIRED_FILES = [
    "frameIndex.xml",
    "frRelation.xml",
    "fulltextIndex.xml",
    "luIndex.xml",
    "semTypes.xml",
]
REQUIRED_DIRS = ["frame", "lu", "fulltext"]


# =========================================================
# LOCATING FRAMENET
# =========================================================
def find_framenet_root(framenet_dir):
    """Return the directory that holds frameIndex.xml inside ``framenet_dir``.

    Only used with the optional --framenet-dir (a local FrameNet copy). Accepts the directory itself, the official ``fndata-1.7`` folder placed
    inside it, or an NLTK data directory (``corpora/framenet_v17``).
    """
    candidates = [
        framenet_dir,
        os.path.join(framenet_dir, "fndata-1.7"),
        os.path.join(framenet_dir, "framenet_v17"),
        os.path.join(framenet_dir, "corpora", "framenet_v17"),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "frameIndex.xml")):
            return path
    return None


def validate_framenet_dir(framenet_dir):
    """Raise a clear error if ``framenet_dir`` is not a usable FrameNet 1.7 copy."""
    if not os.path.isdir(framenet_dir):
        raise FileNotFoundError(
            f"FrameNet directory not found: {framenet_dir}\n"
            "Point --framenet-dir at a FrameNet 1.7 folder, or leave it out to use NLTK's copy."
        )
    root = find_framenet_root(framenet_dir)
    if root is None:
        raise FileNotFoundError(
            f"No frameIndex.xml found in {framenet_dir} (also looked in fndata-1.7/, "
            "framenet_v17/ and corpora/framenet_v17/). Point --framenet-dir at the "
            "folder that contains frameIndex.xml, frame/, lu/ and fulltext/."
        )
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(root, f))]
    missing += [d + "/" for d in REQUIRED_DIRS if not os.path.isdir(os.path.join(root, d))]
    if missing:
        raise FileNotFoundError(
            f"FrameNet directory {root} is incomplete. Missing: {', '.join(missing)}"
        )
    return root


def open_framenet(framenet_dir=None, download=False):
    """Return an NLTK FrameNet 1.7 reader.

    With ``framenet_dir`` the files are read from that directory. Without it,
    NLTK's own ``framenet_v17`` package is used (as in the original script),
    downloading it first if ``download`` is True.
    """
    import nltk
    from nltk.corpus.reader.framenet import FramenetCorpusReader

    if framenet_dir:
        root = os.path.abspath(validate_framenet_dir(framenet_dir))
        # NLTK >= 3.9 only opens corpus folders that are on nltk.data.path
        for path in (root, os.path.dirname(root)):
            if path not in nltk.data.path:
                nltk.data.path.insert(0, path)
        return FramenetCorpusReader(root, REQUIRED_FILES)

    _ensure_nltk(nltk, [("corpora/framenet_v17", "framenet_v17")], download)
    from nltk.corpus import framenet as nltk_fn
    return nltk_fn


def open_wordnet(download=False):
    """Return NLTK's WordNet reader (used to filter lexical units)."""
    import nltk
    _ensure_nltk(nltk, [("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")], download)
    from nltk.corpus import wordnet
    return wordnet


def _ensure_nltk(nltk, packages, download):
    """Download missing NLTK packages (as the original script did) and check they are there."""
    for path, name in packages:
        try:
            nltk.data.find(path)
            continue
        except LookupError:
            pass
        if download:
            print(f"Downloading NLTK package '{name}' ...", flush=True)
            nltk.download(name, quiet=True)
        try:
            nltk.data.find(path)
        except LookupError:
            raise LookupError(
                f"NLTK package '{name}' is not available and could not be downloaded. Check the "
                f"internet connection, or download it manually: python -c \"import nltk; "
                f"nltk.download('{name}')\". Behind an HTTP proxy, set NLTK_ALLOW_PROXIED_URLOPEN=1."
            ) from None


# =========================================================
# POS MAP used while building the lookup tables (original line 32)
# =========================================================
POS_MAP_LOAD = {
    "v": "verb", "verb": "verb",
    "n": "noun", "noun": "noun",
    "a": "adjective", "adj": "adjective", "adjective": "adjective",
    "adv": "adverb", "adverb": "adverb",
    "prep": "preposition",
    "pro": "pronoun",
    "con": "conjunction"
}

# =========================================================
# POS MAP used at generation time (original line 278, overrides the first)
# =========================================================
POS_MAP = {
    "v": "verb", "verb": "verb",
    "n": "noun", "noun": "noun",
    "a": "adjective", "adj": "adjective", "adjective": "adjective",
    "adv": "adverb", "adverb": "adverb",
    "prep": "preposition", "preposition": "preposition",
    "pro": "pronoun", "pron": "pronoun", "pronoun": "pronoun",
    "c": "conjunction", "con": "conjunction", "conjunction": "conjunction"
}

URDU_POS = {
    "verb": "verb",
    "noun": "noun",
    "adjective": "adjective",
    "adverb": "adverb",
    "preposition": "preposition",
    "pronoun": "pronoun",
    "conjunction": "conjunction"
}


def norm(x):
    return str(x).strip().lower()


def clean_fn_text(markup, fallback=""):
    text = re.sub(r"<ex>.*?(</ex>|$)", "\0", markup or "", flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^.!?\0]*:\s*(?=\0)", "", text)
    text = text.replace("\0", " ")
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or re.sub(r"\s+", " ", fallback or "").strip()


def clean_lu_definition(text):
    text = re.sub(r"^\s*((FN|FE|COD|WN)\s*:\s*)+", "", text or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip() or "Definition is not available"


# =========================================================
# LOAD ALL LOOKUP TABLES ONCE
# =========================================================
def load(framenet_dir=None, download=False):
    """Open FrameNet/WordNet and build the lookup tables. Returns the FrameNet reader."""
    global fn, wn, translation_dict, example_dict, lu_dict, lu_def_dict

    fn = open_framenet(framenet_dir, download=download)
    wn = open_wordnet(download=download)

    translation_rows, example_rows, lu_rows = [], [], []

    for frame in fn.frames():
        frame_label = frame.name.replace("_", " ")

        # Frame definition
        translation_rows.append({
            "frame_name": frame_label,
            "input_name": frame_label,
            "voted_translation_eng": clean_fn_text(frame.definitionMarkup, frame.definition)
        })

        # FE definitions
        for fe_name, fe in frame.FE.items():
            translation_rows.append({
                "frame_name": frame_label,
                "input_name": fe_name.replace("_", " "),
                "voted_translation_eng": clean_fn_text(fe.definitionMarkup, fe.definition)
            })

        seen_fes = set()
        for lu_name, lu in frame.lexUnit.items():
            if "." not in lu_name:
                continue

            # Lexical units and their definitions
            lemma, category = lu_name.rsplit(".", 1)
            lu_rows.append({
                "frame_name": frame_label,
                "lexical_unit": lemma,
                "pos": {"c": "conjunction", "pron": "pronoun"}.get(category, category),
                "lu_definition_eng": clean_lu_definition(lu.definition)
            })

            # First annotated example for each FE
            for ex in lu.exemplars:
                for start, end, fe_name in ex.FE[0]:
                    span = ex.text[start:end]
                    if span and fe_name not in seen_fes:
                        seen_fes.add(fe_name)
                        example_rows.append({
                            "frame_name": frame_label,
                            "frame_input": fe_name.replace("_", " "),
                            "voted_translation_eng": ex.text,
                            "voted_span_eng": span
                        })

    df_translation = pd.DataFrame(translation_rows).drop_duplicates(
        subset=["frame_name", "input_name"], keep="first"
    )
    df_examples = pd.DataFrame(example_rows)
    df_lu = pd.DataFrame(lu_rows)
    df_lu_def = df_lu.copy()

    # NORMALIZE ONCE
    for df, cols in [
        (df_translation, ["frame_name", "input_name"]),
        (df_examples, ["frame_name", "frame_input"]),
        (df_lu, ["frame_name", "lexical_unit", "pos"]),
        (df_lu_def, ["frame_name", "lexical_unit", "pos"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str).map(norm)

    df_lu["pos"] = df_lu["pos"].map(POS_MAP_LOAD).fillna(df_lu["pos"])
    df_lu_def["pos"] = df_lu_def["pos"].map(POS_MAP_LOAD).fillna(df_lu_def["pos"])

    # CONVERT TO DICTIONARIES (FASTEST LOOKUP)
    translation_dict = {
        (r.frame_name, r.input_name): r.voted_translation_eng
        for r in df_translation.itertuples()
    }

    example_dict = {
        (r.frame_name, r.frame_input): (r.voted_translation_eng, r.voted_span_eng)
        for r in df_examples.itertuples()
    }

    lu_dict = {}
    for r in df_lu.itertuples():
        lu_dict.setdefault((r.frame_name, r.lexical_unit, r.pos), []).append(r.lexical_unit)

    lu_def_dict = {}
    for r in df_lu_def.itertuples():
        lu_def_dict.setdefault((r.frame_name, r.lexical_unit, r.pos), []).append(r.lu_definition_eng)

    return fn


# =========================================================
# LOOKUPS
# (the "urdu" names are kept from the original multilingual code base;
#  in this English pipeline they return the English FrameNet values)
# =========================================================
def get_cache(cache, key):
    return cache.get(key)


def get_voted_translation(frame_name, input_name):
    return get_cache(
        translation_dict,
        (norm(frame_name), norm(input_name))
    )


def get_voted_translated_examples(frame_name, input_name):
    return example_dict.get(
        (norm(frame_name), norm(input_name)),
        (None, None)
    )


def get_urdu_lexical_units(frame_name, lexical_unit, pos):
    pos = POS_MAP.get(norm(pos), norm(pos))
    return lu_dict.get(
        (norm(frame_name), norm(lexical_unit), pos),
        []
    )


def get_urdu_lexical_units_def(frame_name, lexical_unit, pos):
    pos = POS_MAP.get(norm(pos), norm(pos))
    return lu_def_dict.get(
        (norm(frame_name), norm(lexical_unit), pos),
        []
    )


def to_urdu_pos(pos):
    return URDU_POS.get(POS_MAP.get(str(pos).strip().lower(), str(pos).lower()), pos)


def get_first_sentence(text):
    """
    Heuristic first-sentence extractor.
    Handles English + Urdu better than simple split.
    """

    if pd.isna(text) or not text:
        return "Definition is not available"

    text = str(text).strip().replace("\n", " ")

    # Protect common abbreviations
    protected = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "U.S."]
    for abbr in protected:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?۔؛])\s+', text)

    first = parts[0].replace("<DOT>", ".").strip() if parts else ""

    return first if first else "Definition is not available"


# =========================================================
# LEXICAL UNIT FILTER
# =========================================================
# A lexical unit is used only if WordNet lists its lemma with the same POS.
def _wordnet_pos():
    return {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "adv": wn.ADV}


def valid_lexical_units(frame):
    """Names of the frame's lexical units that are found in WordNet."""
    wordnet_pos = _wordnet_pos()
    valid = []
    for lu_name in frame.lexUnit:
        if lu_name.count(".") != 1:
            continue
        lemma, category = lu_name.split(".")
        if category in wordnet_pos and wn.synsets(lemma, pos=wordnet_pos[category]):
            valid.append(lu_name)
    return valid


def related_frames(name):
    """Names of the frames directly linked to a frame by any relation."""
    frame = fn.frame_by_name(name)
    related = set()
    for rel in frame.frameRelations:
        related.update([rel.get("superFrameName"), rel.get("subFrameName")])
    related.discard(None)
    related.discard(frame.name)
    return related
