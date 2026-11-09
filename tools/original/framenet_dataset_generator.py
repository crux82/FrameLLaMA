import os
import nltk
import re
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn
import pandas as pd
from pickle import NONE
import xml.etree.ElementTree as ET
import random
import csv
from sentence_transformers import SentenceTransformer, util

# Make sure you have downloaded the required corpora
nltk.download('framenet_v17')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Define the folder paths
source_folder = "frames/frame"
target_folder = "Dataset_1/Train/English_1"

# Create the output folders if they do not exist yet
os.makedirs(target_folder, exist_ok=True)

# Helper function → normalize names (remove underscores + extensions + spaces + lowercase)
import pandas as pd

# =========================================================
# POS MAP
# =========================================================
POS_MAP = {
    "v": "verb", "verb": "verb",
    "n": "noun", "noun": "noun",
    "a": "adjective", "adj": "adjective", "adjective": "adjective",
    "adv": "adverb", "adverb": "adverb",
    "prep": "preposition",
    "pro": "pronoun",
    "con": "conjunction"
}

# =========================================================
# LOAD ALL FILES ONCE
# =========================================================
def norm(x):
    return str(x).strip().lower()


# -------------------------
# Load CSV/Excel once
# -------------------------
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


# =========================================================
# NORMALIZE ONCE
# =========================================================
for df, cols in [
    (df_translation, ["frame_name", "input_name"]),
    (df_examples, ["frame_name", "frame_input"]),
    (df_lu, ["frame_name", "lexical_unit", "pos"]),
    (df_lu_def, ["frame_name", "lexical_unit", "pos"]),
]:
    for c in cols:
        df[c] = df[c].astype(str).map(norm)


df_lu["pos"] = df_lu["pos"].map(POS_MAP).fillna(df_lu["pos"])
df_lu_def["pos"] = df_lu_def["pos"].map(POS_MAP).fillna(df_lu_def["pos"])


# =========================================================
# CONVERT TO DICTIONARIES (FASTEST LOOKUP)
# =========================================================
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

# =========================================================
# SAMPLING STRATEGY (T1-T40)
# =========================================================
# Every frame is generated in two passes. Tasks in PARAPHRASED_TASKS use template
# variant 1 in the first pass and variant 2 in the second, with fresh distractors,
# and keep the rows of both passes. All other tasks keep only the first pass,
# because they already produce their full instance count in one pass.
VARIANT = 0

# (Prompt Type, Question ID) as stored in the CSV
PARAPHRASED_TASKS = {
    ("Frame Definition", "1"),         # T1
    ("Frame Definition", "2"),         # T2
    ("Frame Elements", "6"),           # T5
    ("Frame Elements", "5"),           # T7
    ("Frame Elements", "7"),           # T8
    ("Frame Elements", "3"),           # T9
    ("Frame Elements", "4"),           # T10
    ("Semantic Type", "1"),            # T11
    ("Semantic Type", "2"),            # T12
    ("Semantic Type", "3"),            # T13
    ("Lexical Units", "3"),            # T15
    ("Lexical Units", "4"),            # T17
    ("Lexical Unit Polysemy", "1"),    # T18
    ("Lexical Unit Polysemy", "2"),    # T19
    ("Lexical Units", "5"),            # T20
    ("Frame Elements Examples", "1"),  # T21
}


def pick(variants):
    """Template variant for the current pass."""
    return variants[VARIANT % len(variants)]


def merge_passes(first_pass, second_pass):
    """All first-pass rows, plus second-pass rows of paraphrased tasks, in task order."""
    extra = [row for row in second_pass if (row[1], row[2]) in PARAPHRASED_TASKS]
    order = {}
    for row in first_pass + extra:
        order.setdefault((row[1], row[2]), len(order))
    return sorted(first_pass + extra, key=lambda row: order[(row[1], row[2])])


# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cache embeddings
embedding_cache = {}

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
            return c, score   # ✅ break immediately

    return None, None  # if nothing found


def normalize_name(filename):
    name, _ = os.path.splitext(filename)  # remove extension
    return name.replace("_", "").replace(" ", "").lower().strip()


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


def normalize_name(name):
    return (
        str(name)
        .lower()
        .replace(".xml", "")
        .replace(".csv", "")
        .replace("_", "")
        .replace(" ", "")
        .strip()
    )


target_files = [
    f for f in os.listdir(target_folder)
    if f.lower().endswith(".csv")
]

target_names = {
    normalize_name(f)
    for f in target_files
}


all_frames = list(fn.frames())

# =========================================================
# LEXICAL UNIT FILTER
# =========================================================
# A lexical unit is used only if WordNet lists its lemma with the same POS.
WORDNET_POS = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "adv": wn.ADV}


def valid_lexical_units(frame):
    """Names of the frame's lexical units that are found in WordNet."""
    valid = []
    for lu_name in frame.lexUnit:
        if lu_name.count(".") != 1:
            continue
        lemma, category = lu_name.split(".")
        if category in WORDNET_POS and wn.synsets(lemma, pos=WORDNET_POS[category]):
            valid.append(lu_name)
    return valid


# =========================================================
# SEMANTIC NEGATIVE FRAME FILTERING
# =========================================================
# A candidate negative frame Fn is kept for a target frame Ft only if
# sim(A(Ft), A(Fn)) < NEGATIVE_THRESHOLD, where A(F) is the text bundle of F.
NEGATIVE_THRESHOLD = 0.50


def related_frames(name):
    """Names of the frames directly linked to a frame by any relation."""
    frame = fn.frame_by_name(name)
    related = set()
    for rel in frame.frameRelations:
        related.update([rel.get("superFrameName"), rel.get("subFrameName")])
    related.discard(None)
    related.discard(frame.name)
    return related


def frame_bundle(frame):
    """A(F): frame definition, FEs, semantic types, LU definitions and relations."""
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


bundle_embeddings = model.encode(
    [frame_bundle(f) for f in all_frames],
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=True
)
bundle_similarity = util.cos_sim(bundle_embeddings, bundle_embeddings)
frame_positions = {f.name: i for i, f in enumerate(all_frames)}


def negative_frames(name):
    """Frames allowed as negatives for a target frame."""
    target = name.replace(" ", "_")
    excluded = related_frames(target) | {target}
    similarities = bundle_similarity[frame_positions[target]].tolist()
    return [
        f for f, similarity in zip(all_frames, similarities)
        if similarity < NEGATIVE_THRESHOLD and f.name not in excluded
    ]


USE_SINGLE = False

if USE_SINGLE:
    process_frames = [fn.frame_by_name("Abandonment")]
else:
    process_frames = fn.frames()


missing_frames = []

for frame in process_frames:

    # Original NLTK frame name
    nltk_frame_name = frame.name   # e.g. Abounding_with

    # Normalized for comparison
    norm_name = normalize_name(nltk_frame_name)

    # If CSV missing
    if norm_name not in target_names:

        frame_name = nltk_frame_name.replace("_", " ")

        missing_frames.append({
            "frame_name": frame_name,
            "frame_obj": frame
        })


print(f"Total FrameNet frames: {len(all_frames)}")
print(f"Missing frames: {len(missing_frames)}")

for item in missing_frames:

    if item:
        frame_name = item["frame_name"]
        frame_obj = item["frame_obj"]




        # Define the template
        template = {
            "frame_name_questions": [
                {"id": "FDQ1", "text": "Can you describe what typically happens in the \"{frame_name}\" frame? What kind of situation or event does the {frame_name} frame represent?"},
                {"id": "FDQ2", "text": "Can you explain the typical scenario or situation captured by the \"{frame_name}\" frame?"}


            ],
            "frame_name_answers": [
                {"id": "FD1B1", "text": "The \"{frame_name}\" frame represents a type of situation or event in which specific participants (called frame elements) play meaningful roles. It describes a scenario where \"{frame_definition}\". This frame helps structure our understanding of how such events are typically expressed in language, highlighting both the relationships between roles and the underlying semantics of the situation."},
                {"id": "FD1B2", "text": "The \"{frame_name}\" frame is used for situations or events where designated frame elements such as \"{frame_elements}\" participate in key roles. It involves a situation in which \"{frame_definition}\". This helps structure our understanding of how language expresses such occurrences."},




            ],
            "definition_to_frame_questions": [
            {"id": "DFQ1","text": "Based on the description \"{frame_definition}\", identify the frame that best represents this kind of event or situation."},
            {"id": "DFQ2","text": "From the description \"{frame_definition}\", select the frame that captures this event or situation."},

            ],
            "definition_to_frame_answers": [
                {"id": "FD2A1", "text": "The description refers to a structured situation that fits the “{frame_name}” frame because it highlights the core participants and typical actions involved. This frame captures the shared meaning across different contexts where similar events occur, aligning with how language expresses this type of situation."},
                {"id": "FD2A2", "text": "The description matches the \"{frame_name}\" frame as it emphasizes the key participants and actions involved. This frame conveys the common meaning found in various contexts where such events take place, reflecting how language typically represents them."},



            ],
            # New template for Yes/No questions about the frame definition
            "yes_no_template": {
                "questions": [
                    {"id": "FD3Q1", "text": "Does the situation \"{definition}\" accurately represent the typical meaning of the \"{frame_name}\" frame? "},
                    {"id": "FD3Q2", "text": "Is the description \"{definition}\" a correct example of what the \"{frame_name}\" frame usually signifies? "},

                ],
                "answers": [
                    "Yes or No"
                ]
            },

                    "frame_element_questions": [
            {"id": "FE1Q1", "text": "Can you list the key participant roles commonly involved in the situation described by the \"{frame_name}\" frame?"},
            {"id": "FE1Q2", "text": "List the important participant roles usually involved in the scenario captured by the \"{frame_name}\" frame."},

            ],
            "frame_element_answers": [
            {"id": "FE1A1", "text": "The \"{frame_name}\" frame includes a set of core participant roles, also known as frame elements, that define the typical structure of the event. These include roles like \'{frame_elements}\'. These roles capture the essential relationships that organize our understanding of the situation described by the frame."},
            {"id": "FE1A2", "text": "In the \"{frame_name}\" frame, the frame elements are \'{frame_elements}\'. These roles outline the main relationships that shape our understanding of this type of event."},

            ],
            "check_element_questions": [
            {"id": "FE2Q1", "text": "Which frame includes participant roles such as \"{frame_elements}\", representing a situation where someone or something is responsible for causing a significant event?"},
            {"id": "FE2Q2", "text": "Which frame contains participant roles like \"{frame_elements}\" and describes a situation in which someone or something is responsible for bringing about a major event?"},

            ],
            "check_element_answers": [

            {"id": "FE2A1", "text": "The frame that includes roles like \"{frame_elements}\"  is the \"{frame_name}\" frame. These frame elements indicate that the situation involves an intentional or consequential action. Their presence helps define the semantic structure of the event, as understood within the frame."},
            {"id": "FE2A2", "text": "The frame that contains the roles \"{frame_elements}\" is the \"{frame_name}\" frame. These roles signal that the event involves an intentional or consequential action, shaping the semantic structure of the situation."},


            ],
            "multi_selection_question": [

            {"id": "FE3Q1", "text": "Which one of the following roles is typically considered a contributor to the creation of the situation in the \"{frame_name}\" frame?"},
            {"id": "FE3Q2", "text": "Which of the following participant roles in the \"{frame_name}\" frame is usually seen as contributing to the occurrence of the event?"},

        ],
            "multi_selection_question_none": [
            {"id": "FE4Q1","text": "Which of the following roles is not typically involved in causing the event represented by the \"{frame_name}\" frame?"},
            {"id": "FE4Q2","text": "In the context of the \"{frame_name}\" frame, which role is generally not associated with causing the event it describes?"},


            ],
            "yes_no_template_2": {
            "questions": [
                {"id": "FE5Q1","text": "Are roles \"{frame_elements}\" part of the frame elements of Frame \"{frame_name}\"? "},
                {"id": "FE5Q2","text": "In the context of the \"{frame_name}\" frame, does the role \"{frame_elements}\" function as one of the frame elements? "},

                ],
            },
            "check_element_questions_2": [
            {"id": "FE6Q1", "text": "How would you describe the role of the \"{frame_elements}\" when it participates in the context of the \"{frame_name}\" frame? "},
            {"id": "FE6Q2", "text": "In the context of the \"{frame_name}\" frame, the participant role called \"{frame_elements}\" represents a specific function in the event. Can you describe or define in detail what this role involves, what it contributes to the overall meaning of the situation?"},
             ],
            "check_element_answers_2": [
                {"id": "FE6A1", "text": "In the context of the \"{frame_name}\" frame, the role of \"{frame_elements}\" refers \"{definition}\". This frame element represents the agent or initiator of the event. Understanding the role of \"{frame_elements}\" helps define who is responsible for the core action described in this frame."},
                {"id": "FE6A2", "text": "In the \"{frame_name}\" frame, the role of \"{frame_elements}\" is defined as \"{definition}\". This frame element functions as the agent or instigator of the event. Recognizing the \"{frame_elements}\" clarifies who is responsible for carrying out the core action in this scenario."},

                   ],
            "yes_no_template_3": {
            "questions": [
                {"id": "FE7Q1", "text": "Does the definition \"{definition}\" accurately describe the role of \"{frame_element}\" participating in the \"{frame_name}\" frame? "},
                {"id": "FE7Q2", "text": "In the context of the \"{frame_name}\" frame, is the definition \"{definition}\" a correct description of the role played by \"{frame_element}\"? "},

                ],
            },

            "list_laxical_questions": [
                {"id": "LU1Q1", "text": "What are some \'{pos}\' example words or expressions (lexical units) that typically evoke the concept described by the \"{frame_name}\" frame?"},
              {"id": "LU1Q2", "text": "In semantic frame analysis, a lexical unit is a specific word or expression that points to a frame. For the \"{frame_name}\" frame, what are some \'{pos}\' examples of lexical units found in everyday language?"},

            ],
            "list_laxical_answers": [
                {"id": "LU1A1", "text": "Some example words or expressions, known as lexical units, that typically evoke the \"{frame_name}\" frame include \'{pos}\' that convey the core meaning of the situation described in the frame. These lexical units are commonly used in language to activate this specific type of event or experience. For example, words like \'{lexical_units}\' are associated with this frame because they involve the typical roles and actions defined within it."
              },
              {"id": "LU1A2","text": "For the \"{frame_name}\" frame, these are usually \'{pos}\' that describe the main action. Examples include \'{lexical_units}\'. Each of these terms fits the frame because they refer to events, matching the roles and relationships defined in the frame."
              },

            ],
            "all_list_laxical_questions": [
                      {"id": "LU1Q1", "text": "What are some example words or expressions (lexical units) that typically evoke the concept described by the \"{frame_name}\" frame?"
                    },
                    {"id": "LU1Q2", "text": "In semantic frame analysis, a lexical unit is a specific word or expression that points to a frame. For the \"{frame_name}\" frame, what are some examples of lexical units found in everyday language?"
                    }

                    ],
            "all_list_laxical_answers": [
              {"id": "LU1A1", "text": "Some example words or expressions, known as lexical units, that typically evoke the \"{frame_name}\" frame include \'{pos}\' that convey the core meaning of the situation described in the frame. These lexical units are commonly used in language to activate this specific type of event or experience. For example, words like \'{lexical_units}\' are associated with this frame because they involve the typical roles and actions defined within it."
              },
              {"id": "LU1A2","text": "For the \"{frame_name}\" frame, these are usually \'{pos}\' that describe the main action. Examples include \'{lexical_units}\'. Each of these terms fits the frame because they refer to events, matching the roles and relationships defined in the frame."
              }
            ],

              "yes_no_template_4": {
                  "questions": [
              {"id": "LU2Q1", "text": "In the context of the \"{frame_name}\" frame, can the \"{pos}\" form of the word \'{lexical_units}\' be considered a valid lexical unit that evokes this frame? "},
                {"id": "LU2Q2", "text": "Does the \"{pos}\" version of \'{lexical_units}\' function as a lexical unit that activates the \"{frame_name}\" frame? "},
                 ],
              },
            "lexical_unit_def_samples":
                      [
                      {"id": "LU3Q1", "text": "What role does the \'{pos}\' \'{lexical_units}\' play in a sentence that evokes the \"{frame_name}\" frame, and what does it generally mean?"
                      },
                      {"id": "LU3Q2", "text": "How is the \'{pos}\' \'{lexical_units}\' used in sentences that evoke the \"{frame_name}\" frame, and what is its typical meaning?"
                      }
                ],
            "lexical_unit_def_answers": [
                      {"id": "LU3A1","text": "The \'{pos}\' \'{lexical_units}\' is one of the lexical units that evokes the \"{frame_name}\" frame. It typically expresses a specific kind of situation or action represented by this frame. The \'{pos}\' generally means \"{definition}\", aligning with the overall meaning structure of the frame."
                      },
                      {"id": "LU3A2","text": "The \'{pos}\' \'{lexical_units}\' belongs to the set of lexical units linked to the \"{frame_name}\" frame. It conveys an action \"{frame_definition}\". In everyday use, it can describe \"{definition}\" aligning with the core meaning of the frame."
                      }
        ],
            "lexical_unit_def_binary": {
                "questions": [
                      {"id": "LU4Q1","text": "The \'{pos}\' \'{lexical_units}\' is commonly defined as \"{definition}\". Based on this definition, does this lexical unit describe a type of event that corresponds to the kind of situation typically captured by the \"{frame_name}\" frame? Respond with only: Yes or No."
                      },
                        {"id": "LU4Q2","text": "Given that the \'{pos}\' \'{lexical_units}\' is defined as \"{definition}\", would you say that this lexical unit represents an event that aligns with the scenarios usually depicted by the \"{frame_name}\" frame? Please answer with only: Yes or No."
                        }

                      ],
                },
            "lexical_unit_mcq_samples": [
                      {"id": "LUP3Q1","text": "Among the following \'{pos}\', which ones typically trigger the \"{frame_name}\" frame based on their meaning and usage in context?"
                      },
                      {"id": "LUP3Q2","text": "Which of the following \'{pos}\' are most likely to evoke the \"{frame_name}\" frame, considering their meaning and the situations they usually describe?"
                      }
            ],

            "lexical_ambiguity_samples": [
                    {"id": "LUP1Q1", "text": "The \'{pos}\' \'{lexical_units}\' can describe different types of actions or situations depending on context. Based on this, which semantic frames might be evoked by the \'{pos}\' \'{lexical_units}\' across its various meanings?"
                    },
                    {"id": "LUP1Q2", "text": "The \'{pos}\' \'{lexical_units}\' can express different actions or situations depending on context. Which semantic frames could this \'{pos}\' evoke across its range of meanings?"
                    }
                      ],
            "lexical_ambiguity_samples_ans": [
            {"id": "LUP1A1","text": "The \'{pos}\' \'{lexical_units}\' can evoke different semantic frames depending on the context in which it is used, because the same word may refer to distinct conceptual events. Where a single lexical unit maps to multiple frames. For example, \'{lexical_units}\' as {pos} might evoke the \"{frames}\" in broader or metaphorical uses. Each frame captures a different structure of meaning based on the roles involved and the type of situation being described. Understanding which frame is evoked depends on the sentence’s meaning, including which participants are involved and how the action unfolds."
            },
            {"id": "LUP1A2","text": "The word \'{lexical_units}\' can trigger different frames because the same \'{pos}\' can describe distinct conceptual events in language. It can evoke the \"{frames}\". Identifying the right frame requires analyzing the sentence’s meaning, the entities involved, and how the event is structured."
            }

          ],

          "lexical_ambiguity_samples_binary": {
                    "questions": [
                      { "id": "LUP2Q1", "text": "The word \'{lexical_units}\' can also function as a noun in certain contexts. Does this \'{pos}\' usage evoke semantic frames related to actions such as \"{frames}\"? "
                      },
                        {"id": "LUP2Q2", "text": "Can the \'{pos}\' \'{lexical_units}\' evoke frames associated with actions like \"{frames}\"? "
                        }
                    ],
                  },
        "semantic_type_samples": [
                      { "id": "ST1Q1","text": "What broad category of concept or entity is typically represented by the role \"{frame_element}\" in the context of the \'{frame_name}\' frame?"
                      },
                      {"id": "ST1Q2","text": "In the context of the \'{frame_name}\' frame, what general type of concept or entity does the role \"{frame_element}\" usually refer to?"
                      }
        ],

            "semantic_type_samples_ans": [
                      {"id": "ST1A1","text": "The role \"{frame_element}\" in the \'{frame_name}\' frame typically belongs to the broad semantic category of \"{semantic_type}\". Semantic types like these help characterize the nature of participants involved in frames and enable more precise interpretation of how language maps onto real-world situations."
                      },
                      {"id": "ST1A2","text": "In the \'{frame_name}\' frame, the role \"{frame_element}\" is generally classified under the broad semantic category \"{semantic_type}\". Such semantic types help define the nature of participants in a frame and improve our understanding of how language represents real-world situations."
                      }

          ],

            "semantic_type_samples_binary": [
                      {"id": "ST2Q1", "text": "Does the role \"{frame_element}\" in the \'{frame_name}\' frame usually refer to something that can be categorized as an object or concept, such as \"{semantic_type}\"?"
                      },
                        {"id": "ST2Q2", "text": "Is the role \"{frame_element}\" in the \'{frame_name}\' frame typically associated with entities that can be classified as \"{semantic_type}\"?"
                        }
        ],
            "semantic_type_mcq_samples": [
                      {"id": "ST3Q1","text": "What kind of general category does the role \"{frame_element}\" usually refer in frame \'{frame_name}\' ?"
                      },
                      {"id": "ST3Q2","text": "In frame \'{frame_name}\', which overarching category best describes the type of participant or concept referred to by the \"{frame_element}\" role?"
                      }
        ],

        "frame_element_example_samples": [
                {"id": "FEE1Q1", "text": "Can you write a sentence that clearly illustrates a scenario where someone or something is affected as the \'{frame_element}\' in an event of \"{frame_name}\"—demonstrating how the \"{frame_element}\" role is typically expressed within the \'{frame_name}\' frame? (repeat for all available examples)"
                },
                {"id": "FEE1Q2","text": "Could you create a sentence that depicts an event where a person or entity serves as the \"{frame_element}\" in a \"{frame_name}\" scenario, showing how this role is typically realized within the \'{frame_name}\' frame?"
                }
        ],
            "frame_element_example_ans": [
            {"id": "FEE1A1","text": "Here is an example sentence that illustrates the role of the \"{frame_element}\" in the context of the \"{frame_name}\" frame: \"{ex_sentence}\" In this sentence, the role \"{span}\" corresponds to the frame element \'{frame_element}\', defined as \"{fe_definition}\" and the {frame_name} frame evoked by the lexical unit \'{lexical_unit}\'."
            },
            {"id": "FEE1A2","text": "An example illustrating the role of the \"{frame_element}\" in the \"{frame_name}\" frame is: \"{ex_sentence}\" Here the \"{frame_name}\" frame evoked by the lexical unit \'{lexical_unit}\' and the participant  \"{span}\" represents the frame element \'{frame_element}\', defined as \"{fe_definition}\""
            }
        ],
            "frame_element_alignment_prompts": [
                  {"id": "FEE2Q1", "text": "In the sentence \"{ex_sentence}\", does the role of \"{span}\" align with the frame element \"{frame_element}\" as defined in the context of the \"{frame_name}\" frame and the \"{frame_name}\" frame is evoked by the lexical unit \'{lexical_unit}\'?"
                  },
                    {"id": "FEE2Q2","text": "Considering the sentence \"{ex_sentence}\", would you say that the role of \"{span}\" corresponds to the frame element \"{frame_element}\" within the \"{frame_name}\" frame, which is evoked by the lexical unit \'{lexical_unit}\'?"
                    }
        ],
            "frame_inheritance_prompts": [
                      {"id": "IF1Q1","text": "In terms of conceptual relationships between frames, which broader or more general frame does the \"{frame_name}\" frame derive its structure or meaning from through inheritance?"
                      },
                      {"id": "IF1Q2","text": "Considering frame-to-frame relationships, which more general or parent frame does the \"{frame_name}\" frame inherit its structure or meaning from?"
                      }
                  ],
            "frame_inheritance_prompts_ans": [
                      {"id": "IF1A1", "text": "The \"{frame_name}\" frame drawn from a more general frame that captures the core event structure it builds upon. In frame inheritance means that one frame (the child or specific frame) draws its basic organization of meaning and participant roles from another frame (the parent or more general frame). In the case of the \"{frame_name}\" frame, it inherits from the \"{definition_name}\"."
                      },
                      { "id": "IF1A2", "text": "The \"{frame_name}\" frame is based on a more general frame that provides its core event structure. In frame inheritance, a specific frame (child) gets its organization of meaning and roles from a broader frame (parent). The \"{frame_name}\" frame inherits from the \"{definition_name}\" frame."
                      }
            ],
        "frame_inheritance_prompts_binary": {
          "questions": [
                {"id": "IF2Q1", "text": "In frame-to-frame relationships, is the \"{frame_name}\" frame built on top of the \"{definition_name}\" frame where both involve someone doing something to someone else, one participant causes something to happen to another?"
                },
                {"id": "IF2Q2", "text": "In frame-to-frame relationships, does the \"{frame_name}\" frame conceptually derive from the \"{parent_frame}\" frame, with both frames involving an agent causing an action to affect a patient?"
                }
              ],
            },

          "frame_inheritance_prompts_mcq":  [
                      {"id": "IF2Q1","text": "In frame-to-frame relationships, which of the following frames is conceptually derived from the \"{frame_name}\" frame, treating it as the parent that passes down structure and meaning to the child frame?"
                      },
                      {"id": "IF2Q2","text": "In FrameNet’s frame-to-frame relationships, which of the following frames is conceptually derived from the \"{frame_name}\" frame, treating it as the parent that passes down structure and meaning to the child frame?"
                      }
          ],
                "perspective_on_samples": [
                      {"id": "PO1Q1","text": "In frame-to-frame relationships, which frame provides an alternative perspective or interpretive angle on the same basic event described by the \"{frame_name}\" frame, offering a different point of view on the situation?"
                      },
                      {"id": "PO1Q2","text": "In frame-to-frame relationships, which frame offers a different viewpoint or interpretation of the same event represented by the \"{frame_name}\" frame?"
                      }

            ],

            "perspective_on_samples_ans": [

                      {"id": "PO1A1","text":  "\"{definition_name}\" is the broader frame that provides an alternative perspective on the same basic event described in the \"{frame_name}\" frame. This related frame describes the same core event or situation but focuses on it from a different point of view. Such as emphasizing the experience, viewpoint, or role of a specific participant involved in the event."

                      },
                      {"id": "PO1A2","text": "In frame-to-frame relationships, \"{parent_frame}\" This frame offers a different perspective on the same core event as the \"{frame_name}\" frame."
                      }
                     ],

            "perspective_on_samples_binary":[
                      {
                        "id": "PO2Q1","text": "Is there a frame \"{definition_name}\" that describes the same basic situation or event as the \"{frame_name}\" frame, but from a different point of view?"
                      },
                        {
                            "id": "PO2Q2","text": "Does the frame \"{parent_frame}\" represent the same core event as the \"{frame_name}\" frame but from an alternative perspective?"
                        }
                    ],
            "perspective_on_samples_mcq": [
                        {"id": "PO3Q1","text": "Which of the following frames and its definition expresses the same event as the \"{frame_name}\" frame but with different viewpoint?"
                        },
                        {"id": "PO3Q2","text": "Which of the following frames, along with its definition, represents the same event as the \"{frame_name}\" frame but from a different perspective?"
                        }
        ],

            "uses_frame_prompts": [
                  {"id": "UF1Q1","text": "Which frame does the \"{frame_name}\" frame structurally rely on to represent shared actions, roles, or underlying concepts in frame-to-frame relationships?"
                  },
                  {"id": "UF1Q2","text": "In frame-to-frame relationships, which frame provides the structural foundation that the \"{frame_name}\" frame relies on to represent shared roles, actions, or concepts?"
                  }
        ],
            "uses_frame_prompts_ans": [
                  {"id": "UF1A1","text": "The \"{definition_name}\" is structurally relied upon by the \"{frame_name}\" frame to represent shared or foundational actions, roles, or concepts. This structural relationship means that the \"{frame_name}\" frame borrows elements from the \"{definition_name}\" to build more specific meaning."
                  },
                  {"id": "UF1A2","text": "The\"{definition_name}\" provides structural support for the \"{frame_name}\" frame. This means the \"{frame_name}\"  frame borrows shared actions, roles, and concepts from Communication to define more specific events."
                  }
        ],

        "uses_frame_prompts_binary": [
                      {"id": "UF1Q1","text": "Is the concept of {parent_frame} from \"{definition_name}\" frame reused or incorporated into the \"{frame_name}\" frame’s structure?"
                      },
                        {"id": "UF1Q2","text": "Does the \"{frame_name}\" frame structurally incorporate elements from the \"{parent_frame}\" frame?"
                        }
        ],

            "uses_frame_prompts_mcq": [
                    {"id": "UFMCQ1","text": "Which of the following frames provides foundational structural meaning that the \"{frame_name}\" frame builds upon or incorporates in its own representation?"
                    },
                    {"id": "UFMCQ2", "text": "In frame-to-frame relationships, which of the following frames does the \"{frame_name}\" frame use as the structural basis for its own meaning?"
                    }
        ],

            "subframe_open": [
                        {"id": "SF1Q1","text": "What is an example of a smaller, more focused situation (sub-event) that fits within the overall process described by the \"{frame_name}\" frame?"
                        },
                        {"id": "SF1Q2","text": "Can you give an example of a smaller, more specific event that occurs as part of the larger scenario represented by the \"{frame_name}\" frame?"
                        }
            ],
            "subframe_open_ans": [
                  {"id": "SF1A1","text": "The \"{definition_name}\" — a smaller, more specific situation that fits within the overall process described by the \"{frame_name}\" frame — would be one that captures a particular phase or manner of the event. This frame can represent a narrower instance of the broader concept of \"{frame_name}\" frame."
                  },
                  {"id": "SF1A2","text": "Within the broader \"{frame_name}\" frame, the \"{definition_name}\" frame represents a smaller, more detailed event, illustrating a specific aspect of the overall process."
                  }
        ],
            "subframe_open_binary": [
              {"id": "SF2Q1","text": "Is the \"{frame_name}\" frame considered a smaller part of a larger event structure, such as the \"{definition_name}\" frame?"
              },
                {"id": "SF2Q2","text": "Does the \"{parent_frame}\" frame represent a smaller, more specific event that occurs within the broader context of the \"{frame_name}\" frame?"
                }
            ],

          "precedes_ordering": [
                  {"id": "PO1Q1","text": "In terms of event ordering, which any other frame typically happens before \"{frame_name}\" frame in a typical or expected sequence of events?"
                  },
                  {"id": "PO1Q2","text": "In a typical sequence of events, which frame usually occurs before the \"{frame_name}\" frame?"
                  }
        ],

        "precedes_ordering_ans": [
                {"id": "PO1A1","text": "The \"{definition_name}\" frame that typically occurs before the \"{frame_name}\" frame in a usual or expected sequence of events is one that establishes the circumstances or actions leading up to the \"{frame_name}\". This earlier frame describe initial actions or motivations that can logically escalate into the event of Arrest, thereby forming a coherent temporal or causal progression."
                },
                {"id": "PO1A2","text": "The \"{definition_name}\" frame typically comes before the \"{frame_name}\" frame. It represents preparatory actions or motivations that lead into the main event, forming a coherent temporal or causal flow."
                }
        ],
        "precedes_ordering_binary": [
              {"id": "PO2Q1","text": "Does the \"{definition_name}\" frame typically occur before the \"{frame_name}\" frame in a sequence of related events?"
              },
                {"id": "PO2Q2","text": "Is the \"{parent_frame}\" frame considered to happen prior to the \"{frame_name}\" frame in a usual sequence of events?"
                }
        ],

        "Is_Inchoative_prompts": [
                {"id": "II1Q1","text": "Which frame describes the resulting state or change that begins when the event represented by the \"{frame_name}\" frame occurs?"
                },
                {"id": "II1Q2","text": "Which frame captures the outcome or resulting state that starts when the \"{frame_name}\" frame event takes place?"
                }
        ],
        "Is_Inchoative_prompts_ans": [
                {"id": "II1A1","text": "The frame \"{parent_frame}\" represents the resulting state or change that begins when the event described by the \"{frame_name}\" frame occurs. While \"{frame_name}\" focuses on the initiation of an action or process, \"{definition_name}\" captures the ongoing state or condition that follows. This frame-to-frame relationship helps a coherent narrative sequence, showing how an event transitions into its continuation or consequence."
                },
                {"id": "II1A2","text": "The \"{definition_name}\", ” represents the resulting state or ongoing change that follows the event described by the \"{frame_name}\" frame. While the \"{frame_name}\" frame emphasizes the initiation of an action, \"{definition_name}\" captures what happens afterward, forming a coherent temporal sequence."
                }
        ],
            "Is_Inchoative_prompts_binary": [
                {"id": "II2Q1","text": "Does the \"{definition_name}\" frame represent the situation or event that causes the outcome described in the\"{frame_name}\" frame?"
                },
                {"id": "II2Q2","text": "Is the \"{parent_frame}\" frame considered to be the initiating event that leads to the state described in the \"{frame_name}\" frame?"
                }
            ],
        "Is_Causative_prompts": [
                  {"id": "IC1Q1","text": "Which frame represents the effect or outcome that is typically caused by the action described in the \"{frame_name}\" frame?"
                  },
                  {"id": "IC1Q2","text": "Which frame captures the outcome or effect that usually results from the action represented by the \"{frame_name}\" frame?"
                  }
        ],
            "Is_Causative_prompts_ans": [
                {"id": "IC1A1","text": "The frame that represents the effect or outcome typically caused by the action described in the \"{frame_name}\" frame is one that captures the resulting state after the event has occurred. While the \"{frame_name}\" frame focuses on the act of causing \"{parent_frame}\", its causative relationship leads into another frame that describes what follows—such as the \"{definition_name}\" frame. This subsequent frame reflects the outcome or final state, helping to complete the conceptual chain from action to result."
                },
                {"id": "IC1A2","text": "The frame representing the consequence of the \"{frame_name}\" frame highlights the resulting state after the act. The \"{parent_frame}\" showing the effect or final state, establishes a coherent link from the initial action to its outcome."
                }
        ],
        "Is_Causative_prompts_binary": [
                {"id": "IC2Q1","text": "Is the \"{frame_name}\" frame considered the cause that leads to the event structure of the \"{definition_name}\" frame?"
                },
                {"id": "IC2Q2","text": "Does the \"{parent_frame}\" frame represent the initiating action that results in the state described by the \"{frame_name}\" frame?"
                }
        ],
            "Is_Causative_prompts_mcq": [
                      {"id": "IC3Q1","text": "Which of the following frames best represents the outcome or effect caused by the events described in the \"{frame_name}\" frame?"
                      },
                      {"id": "IC3Q2","text": "Which of the following frames captures the typical result or effect that occurs as a consequence of the \"{frame_name}\" frame?"
                      }
        ]
        }





        # --- Helpers --

        def search_frames_by_lu(lu_string):
            """
            Search all frames for a given lexical unit string like 'run.v'
            Returns a list of (frame_name, first_sentence_of_frame_definition) pairs.
            """
            if "." not in lu_string:
                raise ValueError("Input must be in the format 'lemma.pos', e.g. 'run.v'")

            lemma, pos = lu_string.split(".")
            results = []

            # Search for exact LU match
            pattern = rf'^{re.escape(lemma)}\.{pos}$'
            lus = fn.lus(pattern)

            for lu in lus:
                frame = fn.frame(lu['frame']['ID'])  # full frame object
                frame_name = frame.name.replace("_", " ")  # underscores → spaces
                frame_def = frame.definition.strip().replace("\n", " ").replace("_", " ")
                fe_def= get_voted_translation(frame_name.replace("_", " "), frame_name.replace("_", " "))
                first_sentence = get_first_sentence(fe_def)
                # Take only the first sentence

                results.append(f"{frame_name} defined as \"{first_sentence}\"")

            return results



        def clean_definition(def_text: str) -> str:
            """Remove COD:, FN:, etc. and clean spaces."""
            if not def_text:
                return "Definition not found"
            return re.sub(r"^(FN:|COD:)\s*", "", def_text).strip()
        # Fill in the placeholders in the template with actual values and random selection
        def fill_template(frame_data):
            if not frame_data:
                print("No frame data found to process.")
                return None, []

        # Helper function to process lexical units (lemma extraction and capitalization)
        def process_lexical_units(lexical_units):
            processed_units = []
            lemmas = []
            for unit in lexical_units:
                lemma = unit.split('.')[0]  # Extract lemma (everything before the first period)
                pos = unit.split('.')[1]  # Extract POS (everything after the period)
                lemmas.append(pos)
                processed_units.append(lemma.capitalize())  # Capitalize the lemma for display
            return processed_units, lemmas

        # Helper function to randomly select and shuffle frame elements
        def get_shuffled_frame_elements(lexical_units):

          if not lexical_units:
              return ""
          cleaned_units = [
              str(x).strip()
              for x in lexical_units
              if isinstance(x, str) and x.strip()
          ]


          if not cleaned_units:
              return ""

          # Remove duplicates
          cleaned_units = list(dict.fromkeys(cleaned_units))

          # Shuffle
          random.shuffle(cleaned_units)

          selected_units = cleaned_units[:5]

          if len(selected_units) > 1:
              return ", ".join(selected_units[:-1]) + " and " + selected_units[-1]

          return selected_units[0]


        def get_negative_lexical_units_results():

            other_frames = [f.name for f in all_frames if f.name != frame_name]

            negative_samples = []

            for _ in range(1):
                selected_units = ""
                attempt = 0
                while not selected_units and attempt < 10:
                    attempt += 1
                    neg_frame_name = random.choice(other_frames)
                    lus = fn.lus(frame=neg_frame_name)

                    neg_frame_lus_by_pos = {}

                    for lu in lus:
                        # ✅ FIXED extraction
                        lu_name = lu.get('name')
                        if not lu_name or '.' not in lu_name:
                            continue
                        lemma, po = lu_name.rsplit('.', 1)

                        # ❌ Skip invalid ones
                        if not lemma or not po:
                            continue

                        if po not in neg_frame_lus_by_pos:
                            neg_frame_lus_by_pos[po] = []

                        neg_frame_lus_by_pos[po].append(lemma)

                    negative_urdu_units = []
                    for po, lu_list in neg_frame_lus_by_pos.items():
                        for lu in lu_list:
                            urdu_vals = get_urdu_lexical_units(neg_frame_name, lu, po)
                            if urdu_vals:
                                for val in urdu_vals:
                                    if isinstance(val, str) and val.strip():
                                        negative_urdu_units.append(val)

                    if negative_urdu_units:
                        lemma = get_shuffled_frame_elements(negative_urdu_units[:1])
                        return lemma

        # Fill in the placeholders in the template with actual values and random selection
        def italicize(text: str) -> str:
            """
            Wrap a string in Markdown-style asterisks for italics.
            This will appear italic in Markdown viewers or Jupyter Notebook.
            """
            if not isinstance(text, str):
                return text
            text = text.strip()
            if not text:
                return text
            return f"*{text}*"

        def get_random_frame_elements(frame_elements, num_elements):
            shuffled_elements = random.sample(frame_elements, len(frame_elements))
            selected_elements = shuffled_elements[:num_elements]

            # If more than one element is selected, insert 'and' before the last element
            if len(selected_elements) > 1:
                return ', '.join(selected_elements[:-1]) + 'and' + selected_elements[-1]
            else:
                return selected_elements[0]

        def fill_template(frame_data):
            if not frame_data:
                print("No frame data found to process.")
                return None, None  # Ensure it returns None for both frame_name and csv_data

            csv_data = []
            frame_name = None  # Define the frame_name variable outside the loop

            frame_name = frame_data['name']
            frame_nam = frame_name

            # Every negative below is drawn from this filtered pool of frames
            all_frames = negative_frames(frame_name)

            frame = fn.frame(frame_name.replace(" ","_"))
            frame_elements = list(frame.FE.keys())  # all frame elements
            frame_element = get_random_frame_elements(frame_elements, 1)


            print("ID:  Samples")
            print(f"\nQuestion 1.1:")
            question_samples_def = []
            #for question in template['frame_name_questions']:
            question = pick(template['frame_name_questions'])
            print(f"{question['id']}: {question['text'].format(frame_name=frame_name)}")

            print(f"\nAnswers:")
            #for answer in template['frame_name_answers']:
            answer = pick(template['frame_name_answers'])
            frame_def = get_voted_translation(frame_name=frame_name, input_name = frame_name)
            #frame_def = get_first_sentence(definition_transl)
            print(f"{answer['id']}: {answer['text'].format(frame_name=frame_name, frame_elements = frame_element, frame_definition = frame_def)}")
            question_samples_def.append({
                "question": question['text'].format(frame_name=frame_name),
                "id": question['id'],
                "answer": answer['text'].format(frame_name=frame_name, frame_elements = frame_element, frame_definition = frame_def),
                "ids": question['id']
            })
            random.shuffle(question_samples_def)

            for sq in question_samples_def:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Definition",
                  "1",
                  sq['id'],
                  "Opened",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])

            print(f"\nQuestion 1.2:")
            question_samples_def_2 = []
            #for question in template['definition_to_frame_questions']:
            question = pick(template['definition_to_frame_questions'])
            frame_def = get_voted_translation(frame_name=frame_name, input_name = frame_name)
            print(f"{question['id']}: {question['text'].format(frame_definition=frame_def)}")

            print(f"\nAnswers:")
            #for answer in template['definition_to_frame_answers']:
            answer = pick(template['definition_to_frame_answers'])
            print(f"{answer['id']}: {answer['text'].format(frame_name=frame_name, frame_definition=frame_def)}")
            question_samples_def_2.append({
                "question": question['text'].format(frame_definition=frame_def),
                "id": question['id'],
                "answer": answer['text'].format(frame_name=frame_name, frame_definition = frame_def),
                "ids": question['id']
            })
            for sq in question_samples_def_2:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Definition",
                  "2",
                  sq['id'],
                  "Opened",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])

                # Yes/No Questions
            print(f"\nQuestions 1.3:")
            question_samples_fd = []
            # Determine how many questions will use the user-defined variable
            frame_def = get_voted_translation(frame_name=frame_name, input_name = frame_name)
            #for num_questions in template['yes_no_template']['questions']:
            num_questions = random.choice(template['yes_no_template']['questions'])

            question_samples_fd.append({
                "question": num_questions['text'].format(frame_name=frame_name, definition=frame_def),
                "id": num_questions['id'],
                "answer": "Yes",
                "ids": "FD3A1"
            })

        # "No" logic: generate exactly 3 random samples
        #for question_2 in template["yes_no_template"]["questions"]:

            other_frames = [f.name for f in all_frames if f.name != frame_name]
            random.shuffle(other_frames)
            other_frames = other_frames[:200]
            rand_fram, score = get_first_low_similarity_candidate(frame_name, other_frames, threshold=0.5)
            rand_frame = rand_fram.replace("_", " ")
            definition_transl = get_voted_translation(frame_name=rand_frame, input_name = rand_frame)
            frame_def_random = get_first_sentence(definition_transl)

            question_samples_fd.append({
                "question": num_questions['text'].format(frame_name=frame_name, definition=frame_def_random),
                "id": num_questions['id'],
                "answer": "No",
                "ids": "FD3A2"
            })

            random.shuffle(question_samples_fd)

            for sq in question_samples_fd:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Definition",
                  "3",
                  sq['id'],
                  "Closed",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])

            print("\n" + "="*50 + "\n")



            frame_nam = frame_name

            print("ID: Samples")
            print(f"\nQuestion 2.1:")

            ans_samples = []

            chunk_size = 5
            frame = fn.frame(frame_name.replace(" ","_"))
            frame_elements = list(frame.FE.keys())  # all frame elements
            total_elements = len(frame_elements)

            # Split into chunks
            chunks = [
                frame_elements[i:i + chunk_size]
                for i in range(0, total_elements, chunk_size)
            ]

            num_pairs = min(
                len(template["frame_element_questions"]),
                len(template["frame_element_answers"])
            )

            # T4: 2 paraphrases per group of 5 FEs, plus 2 for the full FE list
            for chunk in chunks + [frame_elements]:

                random_elements_str = get_random_frame_elements(chunk, len(chunk))

                for y in range(num_pairs):

                    q_template = template["frame_element_questions"][y]['text']
                    a_template = template["frame_element_answers"][y]['text']
                    q_id = template["frame_element_questions"][y]['id']
                    a_id = template["frame_element_answers"][y]['id']

                    question_text = q_template.format(frame_name=frame_name)
                    answer_text = a_template.format(
                        frame_name=frame_name,
                        frame_elements=random_elements_str
                    )

                    ans_samples.append({
                        "question_id": q_id,
                        "question": question_text,
                        "answer_id": a_id,
                        "answer": answer_text
                    })


            for pairs in ans_samples:
                print(f"{pairs['question_id']}: {pairs['question']}")
                print(f"{pairs['answer_id']}: {pairs['answer']}")


                csv_data.append([
                frame_name,
                "Frame Elements",
                "1",
                pairs['question_id'],
                "Opened",
                pairs['answer_id'],
                pairs['question'],  # Combined question and options
                pairs['answer']  # Correctly access the answer
                ])



            print(f"\nQuestion 2.2:")
            qs_ans_samples = []

            # Ensure we have enough frame elements
            if len(frame_elements) >= 1:
                # Loop through questions and answers together
                for q, a in zip(template['check_element_questions'], template['check_element_answers']):
                    # Select between 1 and all available frame elements
                    num_elements = min(2, len(frame_elements))
                    random_elements_str = get_random_frame_elements(frame_elements, num_elements)

                    qs_ans_samples.append({
                        "question_id": q['id'],
                        "question": q['text'].format(frame_name=frame_name, frame_elements=random_elements_str),
                        "answer_id": a['id'],
                        "answer": a['text'].format(frame_name=frame_name, frame_elements=random_elements_str)
                    })



            # For question 2.2
            print(f"\nQuestion 2.2:")
            for pairs in qs_ans_samples:
              print(f"{pairs['question_id']}: {pairs['question']}")
              print(f"{pairs['answer_id']}: {pairs['answer']}")


              csv_data.append([
                frame_name,
                "Frame Elements",
                "2",
                pairs['question_id'],
                "Opened",
                pairs['answer_id'],
                pairs['question'],  # Combined question and options
                pairs['answer']  # Correctly access the answer
            ])

            else:
                print(f"⚠️ Skipping frame '{frame_name}' because it has fewer than 3 core elements.")

            # For question 2.3
            print(f"\nQuestion 2.3:")
            question_samples = []
            question_samples_none = []
            frames_re = [
                        frame for frame in all_frames
                        if len(frame.FE) >= 1  and frame.name != frame_name
                    ]

            # If no frame elements with definitions exist, skip
            frame = fn.frame(frame_name.replace(" ","_"))
            fe_list = list(frame.FE.keys())  # all frame elements
            if fe_list:
                for correct_opt in fe_list:
                    # Pick random prompt from pool
                    question = pick(template['multi_selection_question'])
                    #print(f"{question['id']}: {question['text'].format(frame_name=frame_name)}")
                    definition_transl = get_voted_translation(frame_name=frame_name, input_name = correct_opt.replace("_"," "))
                    frame_def = get_first_sentence(definition_transl)
                    correct_option= f"{correct_opt} defined as: \"{frame_def}\""

                    frame_two = random.sample(frames_re, min(200, len(frames_re)))
                    candidate_texts = []
                    candidate_meta = []  # keep mapping for later translation
                    for frame_n in frame_two:
                        core_fes = [
                            fe_name for fe_name, fe in frame_n.FE.items()
                            if fe.coreType == "Core" and fe.definition.strip()
                        ]

                        if not core_fes:
                            continue
                        fe_random = random.choice(core_fes)
                        text = f"{frame_n.name} : {fe_random}"
                        candidate_texts.append(text)
                        candidate_meta.append((frame_n.name, fe_random))


                    incorrect_option = []
                    correct_text = correct_opt  # should also be in English form ideally

                    for i, cand_text in enumerate(candidate_texts):
                        rand_fram, score = get_first_low_similarity_candidate(correct_text, [cand_text], threshold=0.4)
                        if rand_fram:
                            incorrect_option.append((candidate_meta[i], score))
                        if len(incorrect_option) == 2:
                            break

                    final_incorrect_options = []

                    for (frame_n, fe_name), score in incorrect_option:
                        definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name=fe_name.replace("_", " "))
                        frame_def_el = get_first_sentence(definition_transl)
                        final_incorrect_options.append(f"{fe_name} defined as \"{frame_def_el}\"")

                    # Options: one correct + incorrects + "None of them"
                    options = final_incorrect_options + [correct_option, "None of Them"]
                    options = random.sample(options, len(options))  # shuffle

                    # Assign letters
                    option_letters = {chr(97 + i): opt for i, opt in enumerate(options)}
                    for letter, opt in option_letters.items():
                        mcq = ""
                        #print(f"{letter}) {opt}")

                    # Find correct answer
                    matched_answer = None
                    for letter, opt in option_letters.items():
                        if opt == correct_option:
                            matched_answer = f"{letter}) {opt}"
                            #ids = "FE3A1"
                            break
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                    "question": question['text'].format(frame_name=frame_name),
                    "id": question['id'],
                    "options": option_letters,  # Make sure this variable is defined
                    "answer": matched_answer,
                    "ids": "FE3A1"
                    })

                    # Generate one sample where "None of them" is correct
                    question = pick(template['multi_selection_question'])
                    #print(f"{question['id']}: {question['text'].format(frame_name=frame_name)}")

                    # Options: only incorrect + "None of them"
                    frame_two = random.sample(frames_re, min(200, len(frames_re)))
                    candidate_texts = []
                    candidate_meta = []  # keep mapping for later translation
                    for frame_n in frame_two:
                        core_fes = [
                            fe_name for fe_name, fe in frame_n.FE.items()
                            if fe.coreType == "Core" and fe.definition.strip()
                        ]

                        if not core_fes:
                            continue
                        fe_random = random.choice(core_fes)
                        text = f"{frame_n.name} : {fe_random}"
                        candidate_texts.append(text)
                        candidate_meta.append((frame_n.name, fe_random))

                    incorrect_option = []
                    correct_text = correct_opt  # should also be in English form ideally
                    for i, cand_text in enumerate(candidate_texts):
                        rand_fram, score = get_first_low_similarity_candidate(correct_text, [cand_text], threshold=0.4)
                        if rand_fram:
                            incorrect_option.append((candidate_meta[i], score))
                        if len(incorrect_option) == 3:
                            break

                    final_incorrect_options = []

                    for (frame_n, fe_name), score in incorrect_option:
                        definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name=fe_name.replace("_", " "))
                        frame_def_el = get_first_sentence(definition_transl)
                        final_incorrect_options.append(f"{fe_name} defined as: \"{frame_def_el}\"")
                    options = final_incorrect_options + ["None of Them"]
                    options = random.sample(options, len(options))  # shuffle

                    # Assign letters
                    option_letters = {chr(97 + i): opt for i, opt in enumerate(options)}
                    for letter, opt in option_letters.items():
                        mcq = ""
                        #print(f"{letter}) {opt}")

                    # Correct answer is always "None of them"
                    matched_answer = None
                    for letter, opt in option_letters.items():
                        if opt == "None of Them":
                            matched_answer = f"{letter}) {opt}"
                            #ids = "FE3A2"
                            break
                    #print(f"Answer: {matched_answer}\n")


                    question_samples_none.append({
                        "question": question['text'].format(frame_name=frame_name),
                        "id": question['id'],
                        "options": option_letters,  # Make sure this variable is defined
                        "answer": matched_answer,
                        "ids":"FE3A1"
                    })
             # Combine the two dictionaries/lists
            all_question_samples = question_samples + question_samples_none

            # For question 2.3
            print(f"\nQuestion 2.3:")
            for selected_sample in all_question_samples:
                # Print the question and options
                print(f"{selected_sample['id']}: {selected_sample['question']}")
                for letter, option in selected_sample['options'].items():
                    print(f"{letter}) {option}")

                print(f"{selected_sample['ids']}: {selected_sample['answer']}\n")

                # Create a string for the question and options
                options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
                combined_question = f"{selected_sample['question']} \n {options_string}"

                # Append the selected question and answer details to csv_data
                csv_data.append([
                    frame_name,
                    "Frame Elements",
                    "3",
                    selected_sample['id'],
                    "MCQ",
                    selected_sample['ids'],
                    combined_question,
                    selected_sample['answer']
                ])


            print("\nQuestion 2.4:")

            question_samples_incorrect = []
            question_samples_incor = []
            # If no frame elements with definitions exist, skip

            if fe_list:
                # Generate one sample for each correct FE
                for correct_opt in fe_list:
                    # Pick random prompt from pool
                    question = pick(template['multi_selection_question_none'])
                    #print(f"{question['id']}: {question['text'].format(frame_name=frame_name)}")
                    frame_two = random.sample(frames_re, min(200, len(frames_re)))
                    candidate_texts = []
                    candidate_meta = []  # keep mapping for later translation
                    for frame_n in frame_two:
                        core_fes = [
                            fe_name for fe_name, fe in frame_n.FE.items()
                            if fe.coreType == "Core" and fe.definition.strip()
                        ]

                        if not core_fes:
                            continue
                        fe_random = random.choice(core_fes)
                        text = f"{frame_n.name} : {fe_random}"
                        candidate_texts.append(text)
                        candidate_meta.append((frame_n.name, fe_random))
                    incorrect_option = []
                    correct_text = correct_opt  # should also be in English form ideally
                    for i, cand_text in enumerate(candidate_texts):
                        rand_fram, score = get_first_low_similarity_candidate(correct_text, [cand_text], threshold=0.4)
                        if rand_fram:
                            incorrect_option.append((candidate_meta[i], score))
                        if len(incorrect_option) == 1:
                            break

                    correct_option = []

                    for (frame_n, fe_name), score in incorrect_option:
                        definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name=fe_name.replace("_", " "))
                        frame_def_el = get_first_sentence(definition_transl)
                        correct_option.append(f"{fe_name} defined as: \"{frame_def_el}\"")

                    frame = fn.frame(frame_name.replace(" ","_"))
                    fe_list = list(frame.FE.keys())
                    if len(fe_list) < 2:
                      frame_two = fe_list
                    else:
                      frame_two = random.sample(fe_list, 2)
                    incorrect_options = []
                    for frame_n in frame_two:
                        definition_transl = get_voted_translation(frame_name=frame_name, input_name = frame_n.replace("_"," ") )
                        frame_def_el = get_first_sentence(definition_transl)
                        incorrect_options.append(f"{frame_n} defined as: \"{frame_def_el}\"")

                    if not correct_option:
                        continue
                    correct_option = correct_option[0]

                    # Options: one correct + incorrects + "None of them"
                    options = incorrect_options + [correct_option, "None of Them"]
                    options = random.sample(options, len(options))  # shuffle

                    # Assign letters
                    option_letters = {chr(97 + i): opt for i, opt in enumerate(options)}
                    for letter, opt in option_letters.items():
                        mcq = " "
                        #print(f"{letter}) {opt}")

                    # Find correct answer
                    matched_answer = None
                    for letter, opt in option_letters.items():
                        if opt == correct_option:
                            matched_answer = f"{letter}) {opt}"
                            ids = "FE4A1"
                            break
                    #print(f"Answer: {matched_answer}\n")

                    question_samples_incorrect.append({
                    "question": question['text'].format(frame_name=frame_name),
                    "id": question['id'],
                    "options": option_letters,  # Make sure this variable is defined
                    "answer": matched_answer,
                    "ids":ids
                    })

                    # Generate one sample where "None of them" is correct
                    question = pick(template['multi_selection_question_none'])
                    #print(f"{question['id']}: {question['text'].format(frame_name=frame_name)}")
                    frame = fn.frame(frame_name.replace(" ","_"))
                    fe_list = list(frame.FE.keys())  # all frame elements
                    frame_name = frame_name.replace("_", " ")
                    if len(fe_list) < 3:
                      frame_two = fe_list
                    else:
                      frame_two = random.sample(fe_list, 3)
                    incorrect_options = []
                    for frame_n in frame_two:
                        definition_transl = get_voted_translation(frame_name=frame_name, input_name = frame_n.replace("_"," ") )
                        frame_def_el = get_first_sentence(definition_transl)
                        incorrect_options.append(f"{frame_n} defined as \"{frame_def_el}\"")

                    options = incorrect_options + ["None of Them"]
                    options = random.sample(options, len(options))  # shuffle

                    # Assign letters
                    option_letters = {chr(97 + i): opt for i, opt in enumerate(options)}
                    for letter, opt in option_letters.items():
                        mcq = ""
                        #print(f"{letter}) {opt}")

                    # Correct answer is always "None of them"
                    matched_answer = None
                    for letter, opt in option_letters.items():
                        if opt == "None of Them":
                            matched_answer = f"{letter}) {opt}"
                            ids = "FE4A2"
                            break
                    #print(f"Answer: {matched_answer}\n")

                    question_samples_incor.append({
                        "question": question['text'].format(frame_name=frame_name),
                        "id": question['id'],
                        "options": option_letters,  # Make sure this variable is defined
                        "answer": matched_answer,
                        "ids":ids
                        })

            print("\n" + "="*50 + "\n")



              # Combine the two dictionaries/lists
            all_question_samples = question_samples_incorrect + question_samples_incor

            # For question 2.4
            print(f"\nQuestion 2.4:")
            for selected_sample in all_question_samples:
                # Print the question and options
                print(f"{selected_sample['id']}: {selected_sample['question']}")
                for letter, option in selected_sample['options'].items():
                    print(f"{letter}) {option}")

                print(f"{selected_sample['ids']}: {selected_sample['answer']}\n")

                # Create a string for the question and options
                options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
                combined_question = f"{selected_sample['question']} \n {options_string}"

                # Append the selected question and answer details to csv_data
                csv_data.append([
                    frame_name,
                    "Frame Elements",
                    "4",
                    selected_sample['id'],
                    "MCQ",
                    selected_sample['ids'],
                    combined_question,
                    selected_sample['answer']
                ])


            print("\n" + "="*50 + "\n")


            frame_nam = frame_name
            # If no frame elements with definitions exist, skip
            frame = fn.frame(frame_name.replace(" ","_"))
            frame_elements = list(frame.FE.keys())  # all frame elements

            print("ID:  Samples")
            #print("\nQuestion 2.5:")
            # Initialize an empty list to hold the question samples
            question_samples = []

            # Iterate to generate 9 questions
            for _ in range(1):
                question_1 = pick(template["yes_no_template_2"]["questions"])
                # "Yes" answer logic
                elements_length_random = random.randint(1, len(frame_elements[:3]))
                selected_elements = random.sample(frame_elements, elements_length_random)  # Select two elements randomly
                selected_elements_str = get_shuffled_frame_elements(selected_elements)
                #selected_elements_str = ', '.join(selected_elements[:-1]) + ' and ' + selected_elements[-1]

                # Append the question and answer pair to the question_samples list
                question_samples.append({
                    "question": question_1['text'].format(frame_name=frame_name, frame_elements=selected_elements_str),
                    "id": question_1['id'],
                    "answer": "Yes",
                    "ids": "FE4A1"
                })

                # Randomly select a question template
                question_1 = pick(template["yes_no_template_2"]["questions"])

                # "No" answer logic

                frames_with_fe = [frame for frame in all_frames if len(frame.FE) > 3]

                # Randomly select one frame from the filtered list
                random_fram = random.sample(frames_with_fe, min(10, len(frames_with_fe)))
                random.shuffle(random_fram)
                fe_random_frame_elements = []
                for random_frame in random_fram:
                  random_frame_elements = list(random_frame.FE.keys())
                  elements_length_random = random_frame_elements[:len(selected_elements)]
                  selected_elements_fe = get_shuffled_frame_elements(elements_length_random)
                  fe_random_frame_elements.append(selected_elements_fe)

                selected_elements_str, score = get_first_low_similarity_candidate(selected_elements_str, fe_random_frame_elements, threshold=0.5)
                # Append the question and answer pair to the question_samples list
                question_samples.append({
                    "question": question_1['text'].format(frame_name=frame_name, frame_elements=selected_elements_str),
                    "id": question_1['id'],
                    "answer": "No",
                    "ids": "FE4A1"
                })
             # Append to CSV data (correctly formatting the answer)
            for sq in question_samples:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Elements",
                  "5",
                  sq['id'],
                  "Closed",
                  sq['ids'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])

            #print("\nQuestions 2.6:")
            frame_element_pair = []
            for i in range(len(frame_elements)):  # Ensure selected_elements is iterable
                print("\n")
                #for index, question in enumerate(template["check_element_questions_2"]):
                selected_element = frame_elements[i]  # Get a unique element for each question

                question = pick(template["check_element_questions_2"])
                answer = pick(template["check_element_answers_2"])
                definition_transl = get_voted_translation(frame_name=frame_name, input_name = selected_element.replace("_"," ") )
                frame_def_el = get_first_sentence(definition_transl)


                    # Store the formatted answer in a dictionary with the key "answers"
                frame_element_pair.append({
                  "question": question['text'].format(frame_name=frame_name, frame_elements=selected_element),
                  "id":question['id'],
                  "answer": answer['text'].format(frame_name=frame_name, frame_elements=selected_element, definition=frame_def_el),
                  "ids":answer['id']
                })

            # Randomly select a question from check_element_questions_2
            for sq in frame_element_pair:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Elements",
                  "6",
                  sq['id'],
                  "Opened",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])



            print("\nQuestions 2.7:")

            question_samples_1 = []
            frames_re = [
                    frame for frame in all_frames
                    if len(frame.FE) >= 1 and len(frame.lexUnit) >= 1 and frame.name != frame_name
                ]
            # "Yes" logic: generate one sample for each frame element
            for fe in frame_elements:
                question_2 =  pick(template["yes_no_template_3"]["questions"])
                definition_transl = get_voted_translation(frame_name=frame_name, input_name = fe.replace("_"," ") )
                frame_def_el = get_first_sentence(definition_transl)

                question_samples_1.append({
                    "question": question_2['text'].format(frame_name=frame_name, frame_element=fe, definition=frame_def_el),
                    "id": question_2['id'],
                    "answer": "Yes",
                    "ids": "FE6A1"
                })

            # "No" logic: generate exactly 3 random samples

                question_2 =  pick(template["yes_no_template_3"]["questions"])

                random.shuffle(frames_re)
                frame_m = random.sample(frames_re, min(200, len(frames_re)))

                candidate_pairs = []
                candidate_texts = []

                for frame_n in frame_m:
                    frame_random = frame_n.name
                    fe_list = list(frame_n.FE.keys())
                    if not fe_list:
                        continue
                    fe_rand = random.choice(fe_list)
                    candidate_pairs.append((frame_random, fe_rand))
                    candidate_texts.append(fe_rand)

                selected_text, score =  selected_elements, score = get_first_low_similarity_candidate(fe, candidate_texts, threshold=0.5)

                selected_frame = None
                selected_fe = None

                if selected_text:
                    for (frame_n, fe_n), text in zip(candidate_pairs, candidate_texts):
                        if fe_n == selected_text:
                            selected_frame = frame_n
                            selected_fe = fe_n
                            break
                if selected_frame and selected_fe:
                    definition_transl = get_voted_translation(
                        frame_name=selected_frame.replace("_", " "),
                        input_name=selected_fe.replace("_", " ")
                    )

                    frame_def_el = get_first_sentence(definition_transl)
                else:
                    frame_def_el = None

                question_samples_1.append({
                    "question": question_2['text'].format(frame_name=frame_name, frame_element=selected_fe , definition=frame_def_el),
                    "id": question_2['id'],
                    "answer": "No",
                    "ids": "FE6A2"
                })



            # Randomly select one sample from the list
            random.shuffle(question_samples_1)

            for sq in question_samples_1:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Frame Elements",
                  "7",
                  sq['id'],
                  "Closed",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])


            print("\nQuestion 3.1:")

            # POS normalization
            POS_MAP = {
                "v": "verb",
                "n": "noun",
                "a": "adjective",
                "adv": "adverb",
                "prep": "preposition",
                "pron": "pronoun",
                "c": "conjunction"
            }

            # Ordered POS
            pos_list = [
                "verb",
                "noun",
                "adverb",
                "adjective",
                "preposition",
                "pronoun",
                "conjunction"
            ]

            # Initialize
            combine_units = {pos: [] for pos in pos_list}

            frame_obj = fn.frame(frame_name.replace(" ", "_"))

            for lu_name in valid_lexical_units(frame_obj):

                if not lu_name or "." not in lu_name:
                    continue

                lemma, category = lu_name.rsplit(".", 1)

                pos = POS_MAP.get(category)

                if pos:
                    combine_units[pos].append(lemma)

            lexical_pairs = []

            for pos in pos_list:

                if not combine_units[pos]:
                    continue

                # Group by 5
                groups = [
                    combine_units[pos][i:i + 5]
                    for i in range(0, len(combine_units[pos]), 5)
                ]

                # T14: 2 paraphrases per group of 5 LUs, plus 2 for all LUs of this POS
                task_groups = [
                    (group, template["list_laxical_questions"], template["list_laxical_answers"])
                    for group in groups
                ]
                task_groups.append((
                    combine_units[pos],
                    template["all_list_laxical_questions"],
                    template["all_list_laxical_answers"]
                ))

                for group, questions, answers in task_groups:

                    urdu_units = []

                    for lemma in group:

                        urdu_vals = get_urdu_lexical_units(
                            frame_name.replace("_", " "),
                            lemma,
                            pos
                        )

                        if urdu_vals:
                            urdu_units.extend(urdu_vals)

                    if not urdu_units:
                        continue

                    urdu_pos = to_urdu_pos(pos)

                    for question, answer in zip(questions, answers):

                        lexical_units_urdu = get_shuffled_frame_elements(
                            urdu_units
                        )

                        lexical_pairs.append({
                            "question": question["text"].format(
                                pos=urdu_pos,
                                frame_name=frame_name
                            ),
                            "id": question["id"],
                            "answer": answer["text"].format(
                                pos=urdu_pos,
                                frame_name=frame_name,
                                lexical_units=lexical_units_urdu
                            ),
                            "ids": answer["id"]
                        })

            for sq in lexical_pairs:
                print(f"{sq['id']}: {sq['question']}")
                print(f"{sq['ids']}: {sq['answer']}")

                csv_data.append([
                frame_name,
                "Lexical Units",
                "1",
                sq['id'],
                "Opened",
                sq['id'],
                sq['question'],  # Combined question and options
                sq['answer']  # Correctly access the answer
              ])


            print("\nQuestions 3.2:")
            question_samples_dic = []

            # Iterate over each POS and process the groups
            for pos in pos_list:
                if pos in combine_units and combine_units[pos]:  # Check if the POS has items
                    lexical_units_list = []
                    for lu_unit in combine_units[pos]:
                        f_name_lu = frame_name.replace("_", " ")
                        urdu_vals = get_urdu_lexical_units(f_name_lu, lu_unit, pos)

                        if not urdu_vals:
                            print(f"⚠️ No match: {frame_name}, {lu_unit}, {pos}")

                        lexical_units_list.extend(urdu_vals)

                    # -------- YES LOGIC: group lexical units in pairs --------
                    yes_groups = [lexical_units_list[i:i + 2] for i in range(0, len(lexical_units_list), 2)]

                    yes_count = 0
                    for group in yes_groups:
                        if len(group) < 2:  # skip if leftover < 2
                            continue
                        question_2 = random.choice(template["yes_no_template_4"]["questions"])
                        selected_units = get_shuffled_frame_elements(group)
                        urdu_pos = to_urdu_pos(pos)
                        question_lu = question_2['text'].format(
                            frame_name=frame_name,
                            lexical_units=selected_units,
                            pos=urdu_pos
                        )
                        question_samples_dic.append({
                            "question": question_lu,
                            "id": question_2['id'],
                            "answer": "Yes",
                            "ids": "LU2A1"
                        })
                        yes_count += 1

                    # -------- NO LOGIC: 1 sample per POS, with a distractor POS tag --------
                    units = random.sample(lexical_units_list, min(2, len(lexical_units_list)))
                    distractor_pos = [
                        p for p in ["verb", "noun", "adjective", "adverb"]
                        if p != pos and not set(units) & {norm(u) for u in combine_units[p]}
                    ]
                    if distractor_pos:
                        question_2 = random.choice(template["yes_no_template_4"]["questions"])
                        question_lu = question_2['text'].format(
                            frame_name=frame_name,
                            lexical_units=get_shuffled_frame_elements(units),
                            pos=to_urdu_pos(random.choice(distractor_pos))
                        )
                        question_samples_dic.append({
                            "question": question_lu,
                            "id": question_2['id'],
                            "answer": "No",
                            "ids": "LU2A2"
                        })



            # Randomly select one sample from the list
            random.shuffle(question_samples_dic)

            for sq in question_samples_dic:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Lexical Units",
                  "2",
                  sq['id'],
                  "Closed",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])



            print("\nQuestions 3.3:")
            lexical_units_pair = []
            frame_obj = fn.frame(frame_name.replace(" ", "_"))
            # Map categories to POS
            pos_map = {
                "v": "verb",
                "n": "noun",
                "a": "adjective",
                "adv": "adverb",
                "prep": "preposition",
                "pron": "pronoun",
                "c": "conjunction"
            }
            for lu_name in valid_lexical_units(frame_obj):

                if not lu_name or "." not in lu_name:
                    continue

                lemm, category = lu_name.rsplit(".", 1)

                pos = POS_MAP.get(category, "unknown")

                if pos == "unknown":
                    continue

                lemma = get_urdu_lexical_units(frame_name, lemm, pos)
                f_name_lu = frame_name.replace("_", " ")
                def_clean = get_urdu_lexical_units_def(f_name_lu, lemm, pos)
                lemma = get_shuffled_frame_elements(lemma)
                question_tmpl = pick(template["lexical_unit_def_samples"])
                answer_tmpl = pick(template["lexical_unit_def_answers"])
                frame_defi = get_voted_translation(frame_name.replace("_"," "), frame_name.replace("_"," "))
                frame_def = get_first_sentence(frame_defi)
                # Format the question
                urdu_pos = to_urdu_pos(pos)
                question_text = question_tmpl['text'].format(
                    frame_name=frame_name,
                    pos=urdu_pos,
                    lexical_units=lemma
                )

                urdu_pos = to_urdu_pos(pos)

                # Format the answer
                answer_text = answer_tmpl['text'].format(
                    frame_name=frame_name,
                    pos=urdu_pos,
                    lexical_units=lemma,
                    definition=def_clean[0] if def_clean else "",
                    frame_definition=frame_def
                )

                #print(f"Q: {question_text}")
                #print(f"A: {answer_text}")

                # Store Q/A pair
                lexical_units_pair.append({
                    "question": question_text,
                    "id": question_tmpl['id'],
                    "answer": answer_text,
                    "ids": answer_tmpl['id']
                })


            for sq in lexical_units_pair:
                  print(f"{sq['id']}: {sq['question']}")
                  print(f"{sq['ids']}: {sq['answer']}")

                  csv_data.append([
                  frame_name,
                  "Lexical Units",
                  "3",
                  sq['id'],
                  "Opened",
                  sq['id'],
                  sq['question'],  # Combined question and options
                  sq['answer']  # Correctly access the answer
                ])


            print("\nQuestions 3.4:")
            lexical_units_pairs = []

            # -------- YES LOGIC: one sample for each lexical unit --------
            for lu_name in valid_lexical_units(frame_obj):

                if not lu_name or "." not in lu_name:
                    continue

                lemm, category = lu_name.rsplit(".", 1)

                pos = POS_MAP.get(category, "unknown")

                if pos == "unknown":
                    continue

                f_name_lu = frame_name.replace("_", " ")
                lemma = get_urdu_lexical_units(f_name_lu, lemm, pos)

                def_clean = get_urdu_lexical_units_def(f_name_lu, lemm, pos)

                lemma = get_shuffled_frame_elements(lemma)
                urdu_pos = to_urdu_pos(pos)

                # Pick random question template
                question_2 = pick(template["lexical_unit_def_binary"]["questions"])

                question_lu = question_2['text'].format(
                    frame_name=frame_name,
                    lexical_units=lemma,
                    pos=urdu_pos,
                    definition=def_clean[0] if def_clean else ""
                )

                lexical_units_pairs.append({
                    "question": question_lu,
                    "id": question_2['id'],
                    "answer": "Yes",
                    "ids": "LU2A1"
                })

                # -------- NO LOGIC: generate 3 samples from random_lexical_units_def --------
                pos = random.choice(["verb", "noun", "adjective", "adverb"])
                question_2 = pick(template["lexical_unit_def_binary"]["questions"])

                frames_re = [
                    frame for frame in all_frames
                    if len(frame.FE) >= 1
                    and len(frame.lexUnit) >= 1
                    and frame.name != frame_name
                ]

                random.shuffle(frames_re)
                frame_m = random.sample(frames_re, min(200, len(frames_re)))

                candidate_pairs = []
                candidate_texts = []

                for frame_n in frame_m:
                    neg_frame_name = frame_n.name

                    lu_items = list(frame_n.lexUnit.keys())
                    if not lu_items:
                        continue

                    # Random lexical unit
                    lu_full = random.choice(lu_items)

                    # Split lemma + POS
                    if "." in lu_full:
                        lu_lemma, lu_pos = lu_full.rsplit(".", 1)
                    else:
                        lu_lemma, lu_pos = lu_full, "unknown"



                    candidate_pairs.append(
                        (neg_frame_name, lu_lemma, lu_pos)
                    )

                    # Similarity BEFORE Urdu conversion
                    candidate_texts.append(lu_lemma)


                selected_text, score = get_first_low_similarity_candidate(
                    lemm,              # correct lexical unit lemma
                    candidate_texts,
                    threshold=0.5
                )
                negative_pairs = []

                if selected_text:
                    for (frame_n, lu_n, pos_n), text in zip(candidate_pairs, candidate_texts):

                        if lu_n == selected_text:

                            negative_frame_name = frame_n
                            negative_lu = lu_n
                            negative_pos = pos_n

                            f_name_lu = negative_frame_name.replace("_", " ")

                            urdu_val = get_urdu_lexical_units(
                                f_name_lu,
                                negative_lu,
                                negative_pos
                            )

                            urdu_def = get_urdu_lexical_units_def(
                                f_name_lu,
                                negative_lu,
                                negative_pos
                            )

                            # Handle list → string
                            if isinstance(urdu_val, list):
                                urdu_val = urdu_val[0] if urdu_val else ""

                            if isinstance(urdu_def, list):
                                urdu_def = urdu_def[0] if urdu_def else ""

                            # Final validation
                            if (
                                isinstance(urdu_val, str)
                                and urdu_val.strip()
                                and isinstance(urdu_def, str)
                                and urdu_def.strip()
                            ):
                                negative_pairs.append({
                                    "frame": negative_frame_name,
                                    "lu": urdu_val.strip(),
                                    "definition": urdu_def.strip(),
                                    "pos": negative_pos
                                })

                            break   # ✅ stop after first valid negative

                selected_unit = ""
                selected_unit_def = ""
                pos_val = ""

                if negative_pairs:

                    selected = random.choice(negative_pairs)

                    selected_unit = selected["lu"]
                    selected_unit_def = selected["definition"]
                    pos_val = selected["pos"]

                else:
                    continue


                urdu_pos = to_urdu_pos(pos)

                question_lu = question_2["text"].format(
                    frame_name=frame_name,
                    lexical_units=selected_unit,
                    pos=urdu_pos,
                    definition=selected_unit_def
                )

                lexical_units_pairs.append({
                    "question": question_lu,
                    "id": question_2["id"],
                    "answer": "No",
                    "ids": "LU2A2"
                })

            for sq in lexical_units_pairs:
                print(f"{sq['id']}: {sq['question']}")
                print(f"{sq['ids']}: {sq['answer']}")

                csv_data.append([
                frame_name,
                "Lexical Units",
                "4",
                sq['id'],
                "Closed",
                sq['id'],
                sq['question'],  # Combined question and options
                sq['answer']  # Correctly access the answer
              ])

            print("\nQuestion 3.5:")

            lexical_units_pairs_mcq = []
            frames_re = [
                f for f in all_frames
                if len(f.lexUnit) >= 3 and f.name != frame_name
            ]

            frame_m = random.sample(
                frames_re,
                k=min(100, len(frames_re))
            )

            negative_pool = []
            sampled_lu = []

            for f in frame_m:

                for lu_name in f.lexUnit.keys():

                    if not lu_name or "." not in lu_name:
                        continue

                    lemma_neg, pos_neg = lu_name.rsplit(".", 1)

                    if not lemma_neg or not pos_neg:
                        continue

                    sampled_lu.append(lemma_neg)
                    negative_pool.append((lemma_neg, pos_neg, f.name))


            frame_lu = fn.frame(frame_name.replace(" ", "_"))

            if frame_lu:

                frame_units = valid_lexical_units(frame_lu)   # FORCE FULL LIST

                for unit in frame_units:

                    if not unit or "." not in unit:
                        continue

                    correct_opt, pos = unit.rsplit(".", 1)

                    # Normalize POS
                    pos_full = {
                        "v": "verb",
                        "n": "noun",
                        "a": "adjective",
                        "adv": "adverb",
                        "prep": "preposition",
                        "pron": "pronoun",
                        "c": "conjunction"
                    }.get(pos, pos)

                    lemma_vals = get_urdu_lexical_units(
                        frame_name.replace("_", " "),
                        correct_opt,
                        pos_full
                    )

                    if not lemma_vals:
                        continue

                    correct_option = get_shuffled_frame_elements(lemma_vals)

                    if not correct_option:
                        continue

                    urdu_pos = to_urdu_pos(pos_full)

                    question = pick(template['lexical_unit_mcq_samples'])

                    incorrect_option = []

                    shuffled_indices = list(range(len(sampled_lu)))
                    random.shuffle(shuffled_indices)

                    for idx in shuffled_indices:

                        cand_text = sampled_lu[idx]

                        rand_lu, score = get_first_low_similarity_candidate(
                            correct_opt,
                            [cand_text],
                            threshold=0.4
                        )

                        if rand_lu:
                            incorrect_option.append(
                                negative_pool[idx]
                            )

                        if len(incorrect_option) >= 5:
                            break

                    final_incorrect_options = []
                    seen = set()

                    for lem, neg_pos, f_name in incorrect_option:

                        urdu_vals = get_urdu_lexical_units(
                            f_name.replace("_", " "),
                            lem,
                            neg_pos
                        )

                        if not urdu_vals:
                            continue

                        urdu_lu = get_shuffled_frame_elements(urdu_vals)

                        if (
                            urdu_lu
                            and urdu_lu != correct_option
                            and urdu_lu not in seen
                        ):
                            final_incorrect_options.append(urdu_lu)
                            seen.add(urdu_lu)

                        if len(final_incorrect_options) == 2:
                            break

                    if len(final_incorrect_options) < 2:
                        continue

                    options = (
                        final_incorrect_options +
                        [correct_option, "None of Them"]
                    )

                    random.shuffle(options)

                    option_letters = {
                        chr(97 + i): opt
                        for i, opt in enumerate(options)
                    }

                    matched_answer = next(
                        (
                            f"{ltr}) {opt}"
                            for ltr, opt in option_letters.items()
                            if opt == correct_option
                        ),
                        None
                    )

                    lexical_units_pairs_mcq.append({
                        "question": question['text'].format(
                            frame_name=frame_name,
                            pos=urdu_pos,
                            lexical_units=correct_option
                        ),
                        "id": question['id'],
                        "options": option_letters,
                        "answer": matched_answer,
                        "ids": "LU5A1"
                    })


                    # =====================================================
                    # NONE VERSION
                    # =====================================================
                    question_none = pick(template['lexical_unit_mcq_samples'])

                    incorrect_option_none = []

                    shuffled_indices = list(range(len(sampled_lu)))
                    random.shuffle(shuffled_indices)

                    for idx in shuffled_indices:

                        cand_text = sampled_lu[idx]

                        rand_lu, score = get_first_low_similarity_candidate(
                            correct_opt,
                            [cand_text],
                            threshold=0.4
                        )

                        if rand_lu:
                            incorrect_option_none.append(
                                negative_pool[idx]
                            )

                        if len(incorrect_option_none) >= 5:
                            break

                    final_none_options = []
                    seen_none = set()

                    for lem, neg_pos, f_name in incorrect_option_none:

                        urdu_vals = get_urdu_lexical_units(
                            f_name.replace("_", " "),
                            lem,
                            neg_pos
                        )

                        if not urdu_vals:
                            continue

                        urdu_lu = get_shuffled_frame_elements(urdu_vals)

                        if (
                            urdu_lu
                            and urdu_lu != correct_option
                            and urdu_lu not in seen_none
                        ):
                            final_none_options.append(urdu_lu)
                            seen_none.add(urdu_lu)

                        if len(final_none_options) == 3:
                            break

                    if len(final_none_options) < 3:
                        continue

                    options_none = final_none_options + ["None of Them"]

                    random.shuffle(options_none)

                    option_letters_none = {
                        chr(97 + i): opt
                        for i, opt in enumerate(options_none)
                    }

                    matched_none = next(
                        (
                            f"{ltr}) {opt}"
                            for ltr, opt in option_letters_none.items()
                            if opt == "None of Them"
                        ),
                        None
                    )

                    lexical_units_pairs_mcq.append({
                        "question": question_none['text'].format(
                            frame_name=frame_name,
                            pos=urdu_pos,
                            lexical_units=correct_option
                        ),
                        "id": question_none['id'],
                        "options": option_letters_none,
                        "answer": matched_none,
                        "ids": "LU5A2"
                    })

            for selected_sample in lexical_units_pairs_mcq:
                # Print the question and options
                print(f"{selected_sample['id']}: {selected_sample['question']}")
                for letter, option in selected_sample['options'].items():
                    print(f"{letter}) {option}")

                print(f"{selected_sample['ids']}: {selected_sample['answer']}\n")

                # Create a string for the question and options
                options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
                combined_question = f"{selected_sample['question']}\n{options_string}"

                # Append the selected question and answer details to csv_data
                csv_data.append([
                    frame_name,
                    "Lexical Units",
                    "5",
                    selected_sample['id'],
                    "MCQ",
                    selected_sample['ids'],
                    combined_question,
                    selected_sample['answer']
                ])
            else:
                print("No lexical unit definitions available.")

            print("\nQuestion 3.7:")
            lexical_units_pair_multi = []
                # Process all lexical units
            frame_lu = fn.frame(frame_name.replace(" ", "_"))

            frame_lus = valid_lexical_units(frame_lu)

            if frame_lus:
              for lu in frame_lus:
              # Split lexical unit into lemma + category (POS)
                  parts = lu.split('.')
                  if len(parts) != 2:
                      continue
                  lemm, category = parts

                  pos_map = {
                      "v": "verb",
                      "n": "noun",
                      "a": "adjective",
                      "adv": "adverb",
                      "prep": "preposition",
                      "pron": "pronoun",
                      "c": "conjunction"
                  }
                  pos = pos_map.get(category, "unknown")
                  if pos == "unknown":
                      continue
                  fe_def= get_voted_translation(frame_name.replace("_", " "), frame_name.replace("_", " "))
                  f_definition = get_first_sentence(fe_def)

                  frames = search_frames_by_lu(lu)
                  lema = get_urdu_lexical_units(frame_name.replace("_", " "), lemm, pos)
                  lemma = get_shuffled_frame_elements(lema)
                  urdu_pos = to_urdu_pos(pos)
                  question_tmpl = pick(template["lexical_ambiguity_samples"])
                  answer_tmpl = pick(template["lexical_ambiguity_samples_ans"])
                  # Format the question
                  question_text = question_tmpl['text'].format(
                      frame_name=frame_name,
                      pos=urdu_pos,
                      lexical_units=lemma,
                      frame_definition=f_definition
                  )
                  # print(f"{question_tmpl['id']}: {question_text}")

                  # Shuffle frames once
                  shuffled_frames = get_shuffled_frame_elements(frames)

                  # Format the answer
                  urdu_pos = to_urdu_pos(pos)
                  answer_text = answer_tmpl['text'].format(
                      frame_name=frame_name,
                      pos=urdu_pos,
                      lexical_units=lemma,
                      frames=shuffled_frames,
                      frame_definition=f_definition
                  )
                  #print(f"Q: {question_text}")
                  #print(f"A: {answer_text}")


                      #print("\nAnswers 3.7:")
                      #print(f"{selected_answer['id']}: {answer_text}\n")

                  # Store Q/A pair
                  lexical_units_pair_multi.append({
                      "question": question_text,
                      "id": question_tmpl['id'],
                      "answer": answer_text,
                      "ids": answer_tmpl['id']
                  })

            for pairs in lexical_units_pair_multi:
              print(f"{pairs['id']}: {pairs['question']}")
              print(f"{pairs['ids']}: {pairs['answer']}")
              print("\n" + "-"*60 + "\n")

              csv_data.append([
              frame_name,
              "Lexical Unit Polysemy",
              "1",
              pairs['id'],
              "Opened",
              pairs['ids'],
              pairs['question'],  # Combined question and options
            pairs['answer']  # Correctly access the answer
            ])

            else:
              print("lexical units not available for this frame")


            print("\nQuestions 3.8:")
            lexical_units_pairs = []

            # -------- YES LOGIC: one sample for each lexical unit --------
            frame_lu = fn.frame(frame_name.replace(" ", "_"))

            frame_lus = valid_lexical_units(frame_lu)


            # Eligible frames only
            random_lu_frames = [
                frame for frame in all_frames
                if len(frame.lexUnit) >= 2 and frame.name != frame_nam
            ]

            # Sample frames directly (shuffle unnecessary)
            sampled_frames = random.sample(
                random_lu_frames,
                k=min(100, len(random_lu_frames))
            )
            # Collect lexical units
            multi_lus = []

            for frame in sampled_frames:

                lu_items = list(frame.lexUnit.keys())   # faster than .values() + .name

                if not lu_items:
                    continue

                for lu_name in lu_items:

                    if not lu_name or "." not in lu_name:
                        continue

                    multi_lus.append(lu_name)

                    # Stop early once 200 reached
                    if len(multi_lus) == 200:
                        break

                if len(multi_lus) == 200:
                    break

            if frame_lus:
              for lu in frame_lus:
              # Split lexical unit into lemma + category (POS)
                parts = lu.split('.')
                if len(parts) != 2:
                    continue
                lemm, category = parts

                # Map categories to POS
                pos_map = {
                    "v": "verb",
                    "n": "noun",
                    "a": "adjective",
                    "adv": "adverb",
                    "prep": "preposition",
                    "pron": "pronoun",
                    "c": "conjunction"
                }
                pos = pos_map.get(category, "unknown")
                if pos == "unknown":
                    continue
                frames = search_frames_by_lu(lu)
                lema = get_urdu_lexical_units(frame_name.replace("_", " "), lemm, pos)
                lemma = get_shuffled_frame_elements(lema)
                urdu_pos = to_urdu_pos(pos)
                # Pick random question template
                question_2 = pick(template["lexical_ambiguity_samples_binary"]["questions"])
                #framess = get_shuffled_frame_elements(frames)
                question_lu = question_2['text'].format(
                    frame_name=frame_name,
                    pos=urdu_pos,
                    lexical_units=lemma,
                    frames=get_shuffled_frame_elements(frames)
                )

                lexical_units_pairs.append({
                    "question": question_lu,
                    "id": question_2['id'],
                    "answer": "Yes",
                    "ids": "LU2A1"
                })

            # -------- NO LOGIC: generate 3 samples from random_lexical_units_def -------

                selected_lu, score = get_first_low_similarity_candidate(
                    lu,
                    multi_lus,
                    threshold=0.5
                )

                frames = search_frames_by_lu(selected_lu) if selected_lu else []

                frames = get_shuffled_frame_elements(frames)
                pos = random.choice(["verb", "noun", "adjective", "adverb", "preposition", "pronoun", "conjunction"])

                #lem, po = lu.rsplit('.', 1)
                #lemm = get_urdu_lexical_units(frame_name.replace("_", " "), lem, po)
                #lu_lemma = get_shuffled_frame_elements(lemm)
    #for question_2 in template["lexical_ambiguity_samples_binary"]["questions"]:

                question_lu = question_2['text'].format(
                    frame_name=frame_name,
                    pos=pos,
                    lexical_units=lemma,
                    frames=frames # Use definition string directly
                )

                lexical_units_pairs.append({
                    "question": question_lu,
                    "id": question_2['id'],
                    "answer": "No",
                    "ids": "LU2A2"
                })


            for pairs in lexical_units_pairs:
              print(f"{pairs['id']}: {pairs['question']}")
              print(f"{pairs['ids']}: {pairs['answer']}")
              print("\n" + "-"*60 + "\n")

              csv_data.append([
              frame_name,
              "Lexical Unit Polysemy",
              "2",
              pairs['id'],
              "Closed",
              pairs['ids'],
              pairs['question'],  # Combined question and options
            pairs['answer']  # Correctly access the answer
            ])
            else:
              print("lexical units not available for this frame")



            print("\nQuestions 4.1:")
            semantic_type_pair = []
            try:
                # Get frame object
                frame = fn.frame(frame_name.replace(" ", "_"))
                clean_frame_name = frame.name.replace("_", " ")

                found_any = False
                for fe_name, fe in frame.FE.items():
                    # Check only Core or Core-Unexpressed with semantic types
                    if fe.coreType in ["Core", "Core-Unexpressed", "Peripheral", "Extra-Thematic", "Other", "Core-Optional", "…"] and fe.semType:

                        clean_fe_name = fe_name.replace("_", " ")

                        # Some FE have multiple semantic types (list)
                        if isinstance(fe.semType, list):
                            sem_types = [st.name.replace("_", " ") for st in fe.semType]
                        else:
                            sem_types = [fe.semType.name.replace("_", " ")]

                        for clean_sem_type in sem_types:
                            # Pick one question template randomly

                            question_tmpl = pick(template["semantic_type_samples"])
                            answer_tmpl = pick(template["semantic_type_samples_ans"])
                            # Format the question
                            question_text = question_tmpl['text'].format(
                                frame_name=clean_frame_name,
                                frame_element=clean_fe_name
                            )
                            # print(f"{question_tmpl['id']}: {question_text}")

                            # Format the answer
                            answer_text = answer_tmpl['text'].format(
                                frame_name=clean_frame_name,
                                frame_element=clean_fe_name,
                                semantic_type=clean_sem_type
                            )

                            print(f"Q: {question_text}")
                            print(f"A: {answer_text}")

                        #print(f"{selected_answer['id']}: {answer_text}\n")

                            # Store Q/A pair
                            semantic_type_pair.append({
                                "question": question_text,
                                "id": question_tmpl['id'],
                                "answer": answer_text,
                                "ids": answer_tmpl['id']
                            })

                            found_any = True

                if not found_any:
                    print("No core frame elements with semantic types found.")

            except LookupError:
                print(f"Error: Frame '{frame_name}' not found in FrameNet.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            for pairs in semantic_type_pair:
              print(f"{pairs['id']}: {pairs['question']}")
              print(f"{pairs['ids']}: {pairs['answer']}")
              print("\n" + "-"*60 + "\n")

              csv_data.append([
              frame_name,
              "Semantic Type",
              "1",
              pairs['id'],
              "Opened",
              pairs['ids'],
              pairs['question'],  # Combined question and options
            pairs['answer']  # Correctly access the answer
            ])



            print("\nQuestions 4.2:")

            semantic_type_pair_binary = []

            try:
                # Get target frame
                frame = fn.frame(frame_name.replace(" ", "_"))
                clean_frame_name = frame.name.replace("_", " ")

                found_any = False

                # Loop over all frame elements
                for fe_name, fe in frame.FE.items():
                    if not fe.semType:  # skip if no semantic type
                        continue

                    clean_fe_name = fe_name.replace("_", " ")

                    # Normalize semantic types into a list
                    if isinstance(fe.semType, list):
                        sem_type_list = [st.name.replace("_", " ") for st in fe.semType if hasattr(st, "name")]
                    elif hasattr(fe.semType, "name"):
                        sem_type_list = [fe.semType.name.replace("_", " ")]
                    else:
                        sem_type_list = []

                    for clean_sem_type in sem_type_list:
                        # ✅ YES SAMPLE
                        selected_question = pick(template["semantic_type_samples_binary"])
                        yes_question = selected_question['text'].format(
                            frame_name=clean_frame_name,
                            frame_element=clean_fe_name,
                            semantic_type=clean_sem_type
                        )
                        semantic_type_pair_binary.append({
                            "question": yes_question,
                            "id": selected_question['id'],
                            "answer": "Yes",
                            "ids": "FS3A2"
                        })

                        #No Samples

                        found_any = True

                        # Correct semantic type only
                        correct_semantic_type = clean_sem_type.strip()

                        neg_fe_name = None
                        neg_sem_type = None
                        negative_frame_name = None
                        attempts = 0

                        all_negative_candidates = []
                        while attempts < 200 and not neg_sem_type:

                            random_frame = random.choice(
                                [f for f in all_frames if f.name != frame.name]
                            )

                            frame_neg_name = random_frame.name.replace("_", " ")

                            for nfe in random_frame.FE.values():

                                if not nfe.semType:
                                    continue

                                # Multiple semantic types
                                if isinstance(nfe.semType, list):

                                    for st in nfe.semType:

                                        if hasattr(st, "name"):

                                            all_negative_candidates.append(
                                                (
                                                    frame_neg_name,
                                                    nfe.name.replace("_", " "),
                                                    st.name.replace("_", " ")
                                                )
                                            )

                                # Single semantic type
                                elif hasattr(nfe.semType, "name"):

                                    all_negative_candidates.append(
                                        (
                                            frame_neg_name,
                                            nfe.name.replace("_", " "),
                                            nfe.semType.name.replace("_", " ")
                                        )
                                    )

                            attempts += 1

                        if all_negative_candidates:

                            candidate_semantic_types = [
                                sem_type
                                for _, _, sem_type in all_negative_candidates
                            ]

                            selected_sem_type, score = get_first_low_similarity_candidate(
                                correct_semantic_type,
                                candidate_semantic_types,
                                threshold=0.5
                            )

                            if selected_sem_type:

                                for frame_cand, fe_cand, sem_cand in all_negative_candidates:

                                    if sem_cand == selected_sem_type:

                                        negative_frame_name = frame_cand
                                        neg_fe_name = fe_cand
                                        neg_sem_type = sem_cand
                                        break

                        if neg_fe_name and neg_sem_type:

                            no_question = selected_question["text"].format(
                                frame_name=clean_frame_name,
                                frame_element=neg_fe_name,
                                semantic_type=neg_sem_type
                            )

                            semantic_type_pair_binary.append({
                                "question": no_question,
                                "id": selected_question["id"],
                                "answer": "No",
                                "ids": "FS3A1"
                            })
                if not found_any:
                    print("No frame elements with semantic types found.")

            except LookupError:
                print(f"Error: Frame '{frame_name}' not found in FrameNet.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            # ✅ Print results
            if semantic_type_pair_binary:
                for pairs in semantic_type_pair_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Semantic Type",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
            else:
                print("No semantic type samples found.")



            print("\nQuestion 4.3:")

            question_samples = []
            try:
                frame = fn.frame(frame_nam.replace(" ", "_"))
                clean_frame_name = frame.name.replace("_", " ")

                found_any = False  # initialize properly

                # ---- Collect semantic types for target frame ----
                frame_semtypes = []
                for fe_name, fe in frame.FE.items():
                    if fe.semType:  # include all FE types
                        if isinstance(fe.semType, list):
                            sem_types = [st.name.replace("_", " ") for st in fe.semType if hasattr(st, "name")]
                        elif hasattr(fe.semType, "name"):
                            sem_types = [fe.semType.name.replace("_", " ")]
                        else:
                            sem_types = []
                        for st in sem_types:
                            frame_semtypes.append((fe_name.replace("_", " "), st))

                if not frame_semtypes:
                    print(f"No semantic types found for frame '{clean_frame_name}'.")
                else:
                    # ✅ Function to collect semantic types from random frames until we reach `needed`
                    def get_neg_semtypes(exclude_semtype, needed=3):
                        collected = []
                        tried_frames = set()

                        while len(collected) < needed:
                            random_frame = random.choice(all_frames)
                            if random_frame.name == frame.name or random_frame.name in tried_frames:
                                continue
                            tried_frames.add(random_frame.name)

                            for nfe in random_frame.FE.values():
                                if nfe.semType:
                                    if isinstance(nfe.semType, list):
                                        for st in nfe.semType:
                                            if hasattr(st, "name"):
                                                cand = st.name.replace("_", " ")
                                                if cand != exclude_semtype and cand not in collected:
                                                    collected.append(cand)
                                    elif hasattr(nfe.semType, "name"):
                                        cand = nfe.semType.name.replace("_", " ")
                                        if cand != exclude_semtype and cand not in collected:
                                            collected.append(cand)

                            # prevent infinite loop if FrameNet is too small
                            if len(tried_frames) > len(all_frames):
                                break

                        return collected[:needed]

                    # ✅ Generate one sample per semantic type
                    for fe_name, correct_semtype in frame_semtypes:
                        question_template = pick(template['semantic_type_mcq_samples'])
                        q_text = question_template['text'].format(frame_name=clean_frame_name, frame_element=fe_name)

                        # Pick 2 incorrect semantic types
                        incorrect_options = get_neg_semtypes(correct_semtype, needed=2)

                        if len(incorrect_options) < 2:
                            continue  # fallback safeguard

                        # Build 4 options
                        options = [correct_semtype] + incorrect_options + ["None of Them"]
                        random.shuffle(options)

                        option_letters = {chr(97+i): opt for i, opt in enumerate(options)}

                        # Correct answer letter
                        correct_letter = [k for k, v in option_letters.items() if v == correct_semtype][0]
                        matched_answer = f"{correct_letter}) {correct_semtype}"

                        # Store YES sample
                        question_samples.append({
                            "question": q_text,
                            "id": question_template['id'],
                            "options": option_letters,
                            "answer": matched_answer,
                            "ids": "FE3A1"
                        })
                        found_any = True

                    # ✅ Generate one “None of them” sample
                    #for question_template in template['semantic_type_mcq_samples']:
                        question_template = pick(template['semantic_type_mcq_samples'])
                        q_text = question_template['text'].format(frame_name=clean_frame_name, frame_element=fe_name)

                        # Pick 3 incorrect semantic types (never include the correct one)
                        incorrect_options = get_neg_semtypes(correct_semtype, needed=3)

                        if len(incorrect_options) < 3:
                            continue  # safeguard

                        options = incorrect_options + ["None of Them"]
                        random.shuffle(options)

                        option_letters = {chr(97+i): opt for i, opt in enumerate(options)}
                        none_letter = [k for k, v in option_letters.items() if v == "None of Them"][0]
                        matched_answer = f"{none_letter}) None of Them"

                        # Store NO sample
                        question_samples.append({
                            "question": q_text,
                            "id": question_template['id'],
                            "options": option_letters,
                            "answer": matched_answer,
                            "ids": "FE3A2"
                        })
                        found_any = True

                    if not found_any:
                        print("No frame elements with semantic types found.")

            except LookupError:
                print(f"Error: Frame '{frame_nam}' not found in FrameNet.")
            except Exception as e:
                print(f"Unexpected error: {e}")

            # ✅ Print results
            if question_samples:
                for selected_sample in question_samples:
                    print(f"{selected_sample['id']}: {selected_sample['question']}")
                    for letter, option in selected_sample['options'].items():
                        print(f"{letter}) {option}")
                    print(f"{selected_sample['ids']}: {selected_sample['answer']}\n")

                    options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
                    combined_question = f"{selected_sample['question']}\n{options_string}"

                    csv_data.append([
                        frame_nam,
                        "Semantic Type",
                        "3",
                        selected_sample['id'],
                        "MCQ",
                        selected_sample['ids'],
                        combined_question,
                        selected_sample['answer']
                    ])
            else:
                print("No semantic type samples found.")


            print("\nQuestion 5.1:")

            frame_element_examples = []
            frame = fn.frame(frame_nam.replace(" ", "_"))
            #print(f"Frame: {frame.name}\nDefinition: {frame.definition}\n{'-'*60}\n")


            # collect FE definitions
            fe_definitions = {fe.name: fe.definition for fe in frame.FE.values()}

            fe_examples = {}

            for lu in frame.lexUnit.values():
                for ex in lu.exemplars:  # annotated examples
                    for fe_group in ex.FE:
                        # normalize into list
                        if isinstance(fe_group, dict):
                            fe_items = [fe_group]
                        elif isinstance(fe_group, (list, tuple)):
                            fe_items = fe_group
                        else:
                            continue

                        for label in fe_items:
                            fe_name, span_text = None, None

                            if isinstance(label, dict):
                                fe_name = label.get('name') or label.get('feName') or label.get('FEName')
                                start, end = label.get('start'), label.get('end')
                                if fe_name and start is not None and end is not None:
                                    span_text = ex.text[start:end]

                            elif isinstance(label, tuple) and len(label) == 3:
                                start, end, fe_name = label
                                span_text = ex.text[start:end]

                            # only collect valid entries
                            if fe_name and span_text:
                                # store also the lexical unit that evoked this example
                                fe_examples.setdefault(fe_name, []).append(
                                    (ex.text, span_text, lu.name)
                                )

            # Print results (FE name + definition + example + LU)
            if fe_examples:
                for fe, examples in fe_examples.items():
                    definition = fe_definitions.get(fe, "Definition is not available")
                    for sent, span, lu_name in examples[:1]:  # show first 3 example
                        question_tmpl = pick(template["frame_element_example_samples"])
                        answer_tmpl = pick(template["frame_element_example_ans"])

                        f_name = frame_nam.replace("_", " ")
                        f_name_lu = frame_nam.replace("_", " ")
                        #print(f"FE: {fe}")
                        #print(f"Definition: {f_name}")
                        sent, fe_span = get_voted_translated_examples(f_name.capitalize(), fe.capitalize().replace("_", " "))
                        fe_def= get_voted_translation(f_name.capitalize(), fe.capitalize().replace("_", " "))
                        fe_definition = get_first_sentence(fe_def)
                        lemma, po = lu_name.rsplit('.', 1)
                        lemma = get_urdu_lexical_units(f_name_lu, lemma, po)
                        lu_clean = get_shuffled_frame_elements(lemma)
                        # Format the question
                        question_text = question_tmpl['text'].format(
                            frame_name=frame_nam.replace("_", " "),
                            fe_definition=fe_definition,
                            frame_element=fe.replace("_", " "),
                            ex_sentence=sent,
                            span=fe_span
                        )
                        # print(f"{question_tmpl['id']}: {question_text}")

                        # Format the answer
                        answer_text = answer_tmpl['text'].format(
                            frame_name=frame_nam.replace("_", " "),
                            fe_definition=fe_definition,
                            frame_element=fe.replace("_", " "),
                            lexical_unit=lu_clean,
                            ex_sentence=sent,
                            span=fe_span
                        )

                        #print(f"Q: {question_text}")
                        #print(f"A: {answer_text}")


                        # Store Q/A pair
                        frame_element_examples.append({
                            "question": question_text,
                            "id": question_tmpl['id'],
                            "answer": answer_text,
                            "ids": answer_tmpl['id']
                        })
            else:
              print(f"No Frame Elements in '{frame_name}' have annotated examples.")

            # ✅ Print results
            if frame_element_examples:
              for pairs in frame_element_examples:
                print(f"{pairs['id']}: {pairs['question']}")
                print(f"{pairs['ids']}: {pairs['answer']}")
                print("\n" + "-"*60 + "\n")

                csv_data.append([
                frame_nam,
                "Frame Elements Examples",
                "1",
                pairs['id'],
                "Opened",
                pairs['ids'],
                pairs['question'],  # Combined question and options
              pairs['answer']  # Correctly access the answer
              ])
            else:
              print("No semantic type samples found.")


              print("\nQuestion 5.2:")

            frame_element_examples_binary = []

            frames_with_min = [
                        frame for frame in all_frames
                        if len(frame.FE) >= 1
                        and len(frame.lexUnit) >= 1
                        and frame.name != frame_nam
                    ]


            if fe_examples and VARIANT == 0:
                for fe, examples in fe_examples.items():
                    definition = fe_definitions.get(fe, "No definition available.")
                    for sent, span, lu_name in examples[:1]:  # only first example per FE

                        # --- YES SAMPLE --
                        f_name = frame_nam.replace("_", " ")
                        f_name_lu = frame_nam.replace("_", " ")
                        sent, fe_span = get_voted_translated_examples(f_name, fe.replace("_", " "))
                        fe_def= get_voted_translation(f_name, fe.replace("_", " "))
                        fe_definition = get_first_sentence(fe_def)
                        lemma, po = lu_name.rsplit('.', 1)
                        lemma = get_urdu_lexical_units(f_name_lu, lemma, po)
                        lu_clean = get_shuffled_frame_elements(lemma)

                        for selected_question in template["frame_element_alignment_prompts"]:

                            yes_question = selected_question['text'].format(
                                frame_name=frame_nam.replace("_", " "),
                                fe_definition=fe_definition,
                                frame_element=fe.replace("_", " "),
                                lexical_unit=lu_clean,
                                ex_sentence=sent,
                                span=fe_span
                            )

                            frame_element_examples_binary.append({
                                "question": yes_question,
                                "id": selected_question['id'],
                                "answer": "Yes",
                                "ids": "FEE2A1"
                            })

                        # --- NO SAMPLE (paired with YES) ---
                    no_question_generated = False


                    # --------------------------------
                    # NO SAMPLE
                    # --------------------------------
                    no_question_generated = False


                    random.shuffle(frames_with_min)
                    frames_with_min_fe = random.sample(frames_with_min, min(200, len(frames_with_min)))

                    candidate_pairs = []
                    candidate_fes = []

                    for frame in frames_with_min_fe:

                        fe_example = {}

                        for lu in frame.lexUnit.values():

                            for ex in lu.exemplars:

                                for fe_group in ex.FE:

                                    fe_items = (
                                        [fe_group]
                                        if isinstance(fe_group, dict)
                                        else fe_group
                                        if isinstance(fe_group, (list, tuple))
                                        else []
                                    )

                                    for label in fe_items:

                                        fe_name, span_text = None, None

                                        # Dict format
                                        if isinstance(label, dict):

                                            fe_name = (
                                                label.get("name")
                                                or label.get("feName")
                                                or label.get("FEName")
                                            )

                                            start, end = label.get("start"), label.get("end")

                                            if fe_name and start is not None and end is not None:
                                                span_text = ex.text[start:end]

                                        # Tuple format
                                        elif isinstance(label, tuple) and len(label) == 3:

                                            start, end, fe_name = label
                                            span_text = ex.text[start:end]

                                        # Valid
                                        if fe_name and span_text:

                                            fe_clean = fe_name.replace("_", " ")

                                            fe_example.setdefault(fe_clean, []).append(
                                                (
                                                    frame.name,
                                                    ex.text,
                                                    span_text,
                                                    lu.name
                                                )
                                            )

                        if fe_example:

                            random_fe = random.choice(list(fe_example.keys()))
                            example = fe_example[random_fe][0]

                            candidate_pairs.append(
                                (
                                    example[0],   # frame_name
                                    random_fe,    # FE
                                    example[1],   # sentence
                                    example[2],   # span
                                    example[3]    # LU
                                )
                            )

                            # Similarity only on FE
                            candidate_fes.append(random_fe)
                    selected_fe, score = get_first_low_similarity_candidate(
                        fe,
                        candidate_fes,
                        threshold=0.5
                    )

                    frame_neg = None
                    sent2 = None
                    span2 = None
                    lu_name2 = None

                    if selected_fe:

                        for frame_name_cand, fe_cand, sent_cand, span_cand, lu_cand in candidate_pairs:

                            if fe_cand == selected_fe:

                                frame_neg = frame_name_cand
                                fe = fe_cand
                                sent2 = sent_cand
                                span2 = span_cand
                                lu_name2 = lu_cand

                                break

                    if frame_neg:

                        f_name = frame_neg.replace("_", " ")

                        # FE translated example
                        sent, fe_span = get_voted_translated_examples(
                            f_name,
                            fe
                        )

                        # FE definition
                        fe_def = get_voted_translation(
                            f_name,
                            fe
                        )

                        fe_definition = get_first_sentence(fe_def)

                        # LU
                        lemma, po = lu_name2.rsplit(".", 1)

                        lemma = get_urdu_lexical_units(
                            f_name,
                            lemma,
                            po
                        )

                        lu_clean = get_shuffled_frame_elements(lemma)

                        no_question = selected_question["text"].format(
                            frame_name=frame_nam.replace("_", " "),
                            fe_definition=fe_definition,
                            frame_element=fe,
                            lexical_unit=lu_clean,
                            ex_sentence=sent,
                            span=fe_span
                        )
                        frame_element_examples_binary.append({
                                "question": no_question,
                                "id": selected_question['id'],
                                "answer": "No",
                                "ids": "FEE2A1"
                            })

                        no_question_generated = True


                        if not no_question_generated:
                            print(f"⚠️ Could not generate a 'No' sample for frame '{frame_nam}'.")
            else:
              print(f"No Frame Elements in '{frame_nam}' have annotated examples.")




            # Now this will always be safe
             # ✅ Print results
            if frame_element_examples_binary:
              for pairs in frame_element_examples_binary:
                print(f"{pairs['id']}: {pairs['question']}")
                print(f"{pairs['ids']}: {pairs['answer']}")
                print("\n" + "-"*60 + "\n")


                csv_data.append([
                frame_nam,
                "Frame Elements Examples",
                "2",
                pairs['id'],
                "Closed",
                pairs['ids'],
                pairs['question'],  # Combined question and options
              pairs['answer']  # Correctly access the answer
              ])
            else:
              print("No semantic type samples found.")


            print("\n" + "="*50 + "\n")

            print("ID: Samples")
            print("\nQuestion 6.1:")
            frame_inherit_examples = []
            try:
                # Load the frame by name
                frame = fn.frame_by_name(frame_name.replace(" ", "_"))

                # Collect unique inherited parent frames (de-duplicated, preserve order)
                parents = []
                seen = set()
                for rel in frame['frameRelations']:
                    if rel['type']['name'] == 'Inheritance':
                        parent = rel.get('superFrameName')
                        if parent and parent != frame['name'] and parent not in seen:
                            seen.add(parent)
                            parents.append(parent)

                if parents:
                    parent_definitions = []
                    parent_frame = []
                    for parent in parents:
                        try:
                            definition_transl = get_voted_translation(frame_name=parent.replace("_", " "), input_name = parent.replace("_", " "))
                            frame_def_fr = get_first_sentence(definition_transl)
                            parent_definitions.append(f"{parent} defined as: \"{frame_def_fr}\"")
                            parent_frame.append(parent)
                        except Exception:
                            parent_definitions.append(f"{parent}: Definition is not available")

                    # Join parents with commas and "and"
                    if len(parent_definitions) > 1:
                        parents_text = ", ".join(parent_definitions[:-1]) + " and " + parent_definitions[-1]

                    else:
                        parents_text = parent_definitions[0]

                    # Number of pairs = min length of questions and answers
                    num_pairs = min(len(template["frame_inheritance_prompts"]),
                                    len(template["frame_inheritance_prompts_ans"]))

                    for i in range(num_pairs):
                        q_template = template["frame_inheritance_prompts"][i]['text']
                        a_template = template["frame_inheritance_prompts_ans"][i]['text']
                        q_id = template["frame_inheritance_prompts"][i]['id']
                        a_id = template["frame_inheritance_prompts_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))
                        for parent_f in parent_frame:
                          #print(parent_f)
                          par_f = get_shuffled_frame_elements(parent_f)
                        answer_text = a_template.format(frame_name=frame_name.replace("_", " "),parent_frame= parent.replace("_", " "), definition_name=parents_text.replace("_", " "))

                        # Store Q/A pair
                        frame_inherit_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })



                    for pairs in frame_inherit_examples:
                        print(f"{pairs['id']}: {pairs['question']}")
                        print(f"{pairs['ids']}: {pairs['answer']}")
                        print("\n" + "-"*60 + "\n")

                        csv_data.append([
                        frame_nam,
                        "Inherit From",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],  # Combined question and options
                    pairs['answer']  # Correctly access the answer
                    ])

                else:
                    print(f"Frame '{frame_nam}' does not inherit from any other frame.")

            except Exception as e:
                print(f"Frame '{frame_nam}' not available. Error: {str(e)}")





            print("\nQuestion 6.2:")
            frame_inherit_examples_binary = []

            # Load the frame by name
            frame = fn.frame_by_name(frame_name.replace(" ", "_"))


            # Collect unique inherited parent frames (de-duplicated, preserve order)
            parents = []
            seen = set()
            for rel in frame['frameRelations']:
                if rel['type']['name'] == 'Inheritance':
                    parent = rel.get('superFrameName')
                    if parent and parent != frame['name'] and parent not in seen:
                        seen.add(parent)
                        parents.append(parent)

            if parents:
                parent_definitions = []
                parent_frame = []
                for parent in parents:
                    try:
                        definition_transl = get_voted_translation(frame_name=parent.replace("_", " "), input_name = parent.replace("_", " "))
                        frame_def_fr = get_first_sentence(definition_transl)

                        parent_definitions.append(f"{parent} defined as: {frame_def_fr}")
                        parent_frame.append(parent)
                    except Exception:
                        parent_definitions.append(f"{parent}: Definition is not available")

                # Join parents with commas and "and"
                if len(parent_definitions) > 1:
                    parents_text = ", ".join(parent_definitions[:-1]) + " and " + parent_definitions[-1]
                else:
                    parents_text = parent_definitions[0]

                questions = template["frame_inheritance_prompts_binary"]["questions"]
                num_pairs = len(questions)

                for i in range(num_pairs):
                    q_template = questions[i]['text']
                    q_id = questions[i]['id']

                    question_text = q_template.format(
                        frame_name=frame_name.replace("_", " "),
                        parent_frame = parent.replace("_", " "),
                        definition_name=parents_text.replace("_", " ")
                    )

                    frame_inherit_examples_binary.append({
                        "question": question_text,
                        "id": q_id,
                        "answer": "Yes",
                        "ids": "FR2A1"
                    })

                    # No Logic

                questions = template["perspective_on_samples_binary"]

                # Remove current frame
                other_frames = [
                    f.name
                    for f in all_frames
                    if f.name != frame_name
                ]

                random.shuffle(other_frames)

                # SAFE group size
                group_size = max(1, len(parents))

                target_groups = 20
                needed_size = target_groups * group_size

                other_frames = other_frames[:needed_size]

                # Group candidates
                frame_groups = [
                    ", ".join(other_frames[j:j + group_size])
                    for j in range(0, len(other_frames), group_size)
                ]

                while len(frame_groups) < target_groups:
                    frame_groups.append("")

                # Perspective string
                parent_frame_string = ", ".join(parents)

                # Similarity
                rand_group, score = get_first_low_similarity_candidate(
                    parent_frame_string.replace("_", " "),
                    frame_groups,
                    threshold=0.5
                )

                # Default values
                random_frame_name = "None"
                parents_text = "Definition is not available"

                # Recover one frame
                if rand_group and isinstance(rand_group, str):

                    candidate_frames = [
                        x.strip()
                        for x in rand_group.split(",")
                        if x.strip()
                    ]

                    selected_frame = (
                        random.choice(candidate_frames)
                        if candidate_frames else None
                    )

                    if selected_frame:
                        try:
                            random_parent_frame = fn.frame_by_name(
                                selected_frame.replace(" ", "_")
                            )

                            random_frame_name = (
                                random_parent_frame.name.replace("_", " ")
                            )

                            definition_transl = get_voted_translation(
                                frame_name=random_frame_name,
                                input_name=random_frame_name
                            )

                            if definition_transl:
                                frame_def_fr = get_first_sentence(definition_transl)

                                parents_text = (
                                    f"{random_frame_name} defined as: {frame_def_fr}"
                                )

                        except Exception:
                            parents_text = f"{random_frame_name}: Definition is not available"

                for question in questions:

                    q_template = question["text"]
                    q_id = question["id"]

                    question_text = q_template.format(
                        frame_name=frame_nam.replace("_", " "),
                        parent_frame=random_frame_name,
                        definition_name=parents_text
                    )

                    print(question_text)

                    frame_inherit_examples_binary.append({
                        "question": question_text,
                        "id": q_id,
                        "answer": "No",
                        "ids": "FR2A2"
                    })


             #Print results
            for pairs in frame_inherit_examples_binary:
                print(f"{pairs['id']}: {pairs['question']}")
                print(f"{pairs['ids']}: {pairs['answer']}")
                print("\n" + "-"*60 + "\n")
                csv_data.append([
                frame_nam,
                "Inherit From",
                "2",
                pairs['id'],
                "Closed",
                pairs['ids'],
                pairs['question'],  # Combined question and options
              pairs['answer']  # Correctly access the answer
              ])


            else:
                print(f"Frame '{frame_name}' does not inherit from any other frame.")



            print("\nQuestion 6.3:")
            question_samples = []
            question_samples_none = []


            # --- Config ---
            frame_name = frame_nam

            # --- Target frame ---
            frame = fn.frame_by_name(frame_name.replace(" ", "_"))

            # --- Collect true parents via Inheritance (de-duplicated, and ensure sub = target) ---
            parents = []
            seen = set()
            for rel in frame.frameRelations:
                try:
                    if rel["type"]["name"] == "Inheritance":
                        super_name = rel.get("superFrameName") or rel.get("superFrame", {}).get("name")
                        sub_name   = rel.get("subFrameName")   or rel.get("subFrame",   {}).get("name")
                        if super_name and (not sub_name or sub_name == frame.name) and super_name != frame.name:
                            if super_name not in seen:
                                seen.add(super_name)
                                parents.append(super_name)
                except KeyError:
                    continue

            # --- Build MCQs (3 options): correct parent + one random distractor + "None of them" ---
            question_samples = []

            if parents:
                # pick ONE correct parent at random (if there are multiple)
                correct_parent = random.choice(parents)
                parent_label = correct_parent.replace("_", " ")
                parent_relations = related_frames(correct_parent)
                correct_p = frame.name.replace("_", " ")
                definition_transl = get_voted_translation(frame_name=correct_p, input_name = correct_p)
                first_sentence = get_first_sentence(definition_transl)
                correct_option_text = f"{correct_p} — {first_sentence}"


                all_names = [f.name for f in all_frames if f.name not in parent_relations]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,2)
                incorrect_option_text = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_option_text.append(f"{frame_n} — {first_sentence}")


                # loop over your templates (expects template['frame_inheritance_prompts_mcq'])
                for q in template['frame_inheritance_prompts_mcq']:
                    options = [correct_option_text, *incorrect_option_text, "None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == correct_option_text)
                    matched_answer = f"{correct_letter}) {option_letters[correct_letter]}"
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=parent_label),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answer,
                        "ids": "PF3A1"
                    })


                # No parents → "None of them" is correct; use two random incorrect frames
                all_names = [f.name for f in all_frames if f.name not in parent_relations]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,3)
                incorrect_options = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_options.append(f"{frame_n} — {first_sentence}")


                for q in template['frame_inheritance_prompts_mcq']:
                    options = incorrect_options + ["None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == "None of Them")
                    matched_answer = f"{correct_letter}) None of Them"
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=parent_label),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answer,
                        "ids": "PF3A2"
                    })

            for selected_sample in question_samples:
              # Print the question and options
              print(f"{selected_sample['id']}: {selected_sample['question']}")
              for letter, option in selected_sample['options'].items():
                  print(f"{letter}) {option}")

              print(f"{selected_sample['ids']}: {selected_sample['answer']}")
              print("\n" + "-"*60 + "\n")

              # Create a string for the question and options
              options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
              combined_question = f"{selected_sample['question']}\n{options_string}"
              # Append the selected question and answer details to csv_data
              csv_data.append([
                  frame_name,
                  "Inherit from",
                  "3",
                  selected_sample['id'],
                  "MCQ",
                  selected_sample['ids'],
                  combined_question,
                  selected_sample['answer']
              ])
            else:
                print(f"Frame '{frame_name}' does not inherit from any other frame.")




            print("\nQuestion 6.4:")

            perspective_examples = []

            try:
                # Load the frame by name
                frame = fn.frame_by_name(frame_name.replace(" ", "_"))

                # Collect perspective-on relations
                perspectives = []
                seen = set()
                for rel in frame['frameRelations']:
                    if rel['type']['name'] == 'Perspective_on':
                        neutral = rel.get('superFrameName')
                        perspectivized = rel.get('subFrameName')

                        if frame_name.replace(" ", "_") in (neutral, perspectivized):
                            other = neutral if frame_name.replace(" ", "_") == perspectivized else perspectivized
                            if other and other not in seen:
                                seen.add(other)
                                perspectives.append(other)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            other_frame = fn.frame_by_name(other.replace(" ", "_"))

                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            frame_def_fr = get_first_sentence(definition_transl)

                            perspective_defs.append(f"{other} defined as: {frame_def_fr}")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other}: Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["perspective_on_samples"]),
                                    len(template["perspective_on_samples_ans"]))

                    for i in range(num_pairs):
                        q_template = template["perspective_on_samples"][i]['text']
                        a_template = template["perspective_on_samples_ans"][i]['text']
                        q_id = template["perspective_on_samples"][i]['id']
                        a_id = template["perspective_on_samples_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        # Pick shuffled elements from perspective frames
                        par_f = []
                        for p_frame in perspective_frame_names:
                            other = p_frame.replace(" ", "_")
                            fes = fn.frame_by_name(other)['FE']
                            shuffled_fes = list(fes.keys())
                            random.shuffle(shuffled_fes)
                            par_f.extend(shuffled_fes[:2])  # just take 2 FEs as example

                        if not par_f:
                            par_f = ["[Frame element is not avialable"]

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Perspective On",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")


            print("\nQuestion 6.5:")
            perspective_on_binary = []

            try:
                # Use a consistent variable name
                frame = fn.frame_by_name(frame_nam.replace(" ", "_"))

                # Collect perspective-on relations
                perspectives = []
                seen = set()
                for rel in frame['frameRelations']:
                    if rel['type']['name'] == 'Perspective_on':
                        neutral = rel.get('superFrameName')
                        perspectivized = rel.get('subFrameName')

                        if frame_name.replace(" ", "_") in (neutral, perspectivized):
                            other = neutral if frame_name.replace(" ", "_") == perspectivized else perspectivized
                            if other and other not in seen:
                                seen.add(other)
                                perspectives.append(other)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            frame_def_fr = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: {frame_def_fr}")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other}: Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["perspective_on_samples_binary"])

                    for i in range(num_pairs):
                        q_template = template["perspective_on_samples_binary"][i]['text']
                        q_id = template["perspective_on_samples_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No Logic

                    questions = template["perspective_on_samples_binary"]

                    # Remove current frame
                    other_frames = [
                        f.name
                        for f in all_frames
                        if f.name != frame_name
                    ]

                    random.shuffle(other_frames)

                    # SAFE group size
                    group_size = max(1, len(perspectives))

                    target_groups = 20
                    needed_size = target_groups * group_size

                    other_frames = other_frames[:needed_size]

                    # Group candidates
                    frame_groups = [
                        ", ".join(other_frames[j:j + group_size])
                        for j in range(0, len(other_frames), group_size)
                    ]

                    while len(frame_groups) < target_groups:
                        frame_groups.append("")

                    # Perspective string
                    parent_frame_string = ", ".join(perspectives)

                    # Similarity
                    rand_group, score = get_first_low_similarity_candidate(
                        parent_frame_string.replace("_", " "),
                        frame_groups,
                        threshold=0.5
                    )

                    # Default values
                    random_frame_name = "None"
                    parents_text = "Definition is not available"

                    # Recover one frame
                    if rand_group and isinstance(rand_group, str):

                        candidate_frames = [
                            x.strip()
                            for x in rand_group.split(",")
                            if x.strip()
                        ]

                        selected_frame = (
                            random.choice(candidate_frames)
                            if candidate_frames else None
                        )

                        if selected_frame:
                            try:
                                random_parent_frame = fn.frame_by_name(
                                    selected_frame.replace(" ", "_")
                                )

                                random_frame_name = (
                                    random_parent_frame.name.replace("_", " ")
                                )

                                definition_transl = get_voted_translation(
                                    frame_name=random_frame_name,
                                    input_name=random_frame_name
                                )

                                if definition_transl:
                                    frame_def_fr = get_first_sentence(definition_transl)

                                    parents_text = (
                                        f"{random_frame_name} defined as: \"{frame_def_fr}\""
                                    )

                            except Exception:
                                parents_text = f"{random_frame_name}: Definition is not available"

                    for question in questions:

                        q_template = question["text"]
                        q_id = question["id"]

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame=random_frame_name,
                            definition_name=parents_text
                        )

                        print(question_text)

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "No",
                            "ids": "FR2A2"
                        })

                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "Perspective On",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])

                else:
                    print(f"Frame '{frame_name}' does not have any Perspective_on relations.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")



            print("\nQuestion 6.6:")
            question_samples = []
            question_samples_none = []


            # --- Config ---
            frame_name = frame_nam

            # --- Target frame ---
            frame = fn.frame_by_name(frame_name.replace(" ", "_"))

            # --- Collect true parents via Inheritance (de-duplicated, and ensure sub = target) ---
            parents = []
            seen = set()
            for rel in frame.frameRelations:
                try:
                  if rel.type.name == "Perspective_on":
                    neutral = rel.superFrameName
                    perspectivized = rel.subFrameName

                    if frame_name.replace(" ", "_") in (neutral, perspectivized):
                        other = neutral if frame_name.replace(" ", "_") == perspectivized else perspectivized
                        if other and other not in seen:
                            seen.add(other)
                            parents.append(other)
                except KeyError:
                    continue

            # --- Build MCQs (3 options): correct parent + one random distractor + "None of them" ---
            question_samples = []

            if parents:
                # pick ONE correct parent at random (if there are multiple)
                correct_parent = random.choice(parents)
                correct_p = correct_parent.replace("_", " ")
                definition_transl = get_voted_translation(frame_name=correct_p, input_name = correct_p)
                first_sentence = get_first_sentence(definition_transl)
                correct_option_text = f"{correct_p} — {first_sentence}"

                all_names = [f.name for f in all_frames if f.name != frame.name]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,2)
                incorrect_option_text = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_option_text.append(f"{frame_n} — {first_sentence}")


                # loop over your templates (expects template['frame_inheritance_prompts_mcq'])
                for q in template['perspective_on_samples_mcq']:
                    options = [correct_option_text, *incorrect_option_text, "None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == correct_option_text)
                    matched_answer = f"{correct_letter}) {option_letters[correct_letter]}"
                    matched_answe = matched_answer.replace("_", " ")
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answe,
                        "ids": "PF3A1"
                    })


                all_names = [f.name for f in all_frames if f.name != frame.name]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,2)
                incorrect_options = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_options.append(f"{frame_n} — {first_sentence}")


                for q in template['perspective_on_samples_mcq']:
                    options = incorrect_options + ["None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == "None of Them")
                    matched_answer = f"{correct_letter}) None of Them"
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answer,
                        "ids": "PF3A2"
                    })

            for selected_sample in question_samples:
              # Print the question and options
              print(f"{selected_sample['id']}: {selected_sample['question']}")
              for letter, option in selected_sample['options'].items():
                  print(f"{letter}) {option}")

              print(f"{selected_sample['ids']}: {selected_sample['answer']}")
              print("\n" + "-"*60 + "\n")

              # Create a string for the question and options
              options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
              combined_question = f"{selected_sample['question']}\n{options_string}"
              # Append the selected question and answer details to csv_data
              csv_data.append([
                  frame_name,
                  "Perspective On",
                  "3",
                  selected_sample['id'],
                  "MCQ",
                  selected_sample['ids'],
                  combined_question,
                  selected_sample['answer']
              ])
            else:
                print(f"Frame '{frame_name}' does not have any Perspective_on relations.")


            print("\nQuestion 6.7:")

            perspective_examples = []
            try:
                # Load the frame by name
                perspectives = []
                f = fn.frame(frame_nam.replace(" ", "_"))
                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]
                    parent = fr["superFrameName"]
                    child = fr["subFrameName"]

                    if rel_type == "Using" and child == frame_name.replace(" ", "_"):  #
                        perspectives.append(parent)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: {first_sentence}")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["uses_frame_prompts"]),
                                    len(template["uses_frame_prompts_ans"]))

                    for i in range(num_pairs):
                        q_template = template["uses_frame_prompts"][i]['text']
                        a_template = template["uses_frame_prompts_ans"][i]['text']
                        q_id = template["uses_frame_prompts"][i]['id']
                        a_id = template["uses_frame_prompts_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Uses",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])

                else:
                    print(f"Frame '{frame_name}' does not have any Uses relations.")
            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")




            print("\nQuestion 6.8:")
            perspective_on_binary = []
            perspectives = []
            try:
                # Use a consistent variable name
                f = fn.frame(frame_nam.replace(" ", "_"))
                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]
                    parent = fr["superFrameName"]
                    child = fr["subFrameName"]

                    if rel_type == "Using" and child == frame_name.replace(" ", "_"):  # Arrest is the child
                        perspectives.append(parent)

                if perspectives:
                    perspective_defs = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)

                            perspective_defs.append(f"{first_sentence}")
                        except Exception:
                            perspective_defs.append(f"Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["uses_frame_prompts_binary"])

                    for i in range(num_pairs):
                        q_template = template["uses_frame_prompts_binary"][i]['text']
                        q_id = template["uses_frame_prompts_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No Logic


                    questions = template["uses_frame_prompts_binary"]

                    # Remove current frame
                    other_frames = [
                        f.name
                        for f in all_frames
                        if f.name != frame_name
                    ]

                    random.shuffle(other_frames)

                    # SAFE group size
                    group_size = max(1, len(perspectives))

                    target_groups = 20
                    needed_size = target_groups * group_size

                    other_frames = other_frames[:needed_size]

                    # Group candidates
                    frame_groups = [
                        ", ".join(other_frames[j:j + group_size])
                        for j in range(0, len(other_frames), group_size)
                    ]

                    while len(frame_groups) < target_groups:
                        frame_groups.append("")

                    # Perspective string
                    parent_frame_string = ", ".join(perspectives)

                    # Similarity
                    rand_group, score = get_first_low_similarity_candidate(
                        parent_frame_string.replace("_", " "),
                        frame_groups,
                        threshold=0.5
                    )

                    # Default values
                    random_frame_name = "None"
                    parents_text = "Definition is not available"

                    # Recover one frame
                    if rand_group and isinstance(rand_group, str):

                        candidate_frames = [
                            x.strip()
                            for x in rand_group.split(",")
                            if x.strip()
                        ]

                        selected_frame = (
                            random.choice(candidate_frames)
                            if candidate_frames else None
                        )

                        if selected_frame:
                            try:
                                random_parent_frame = fn.frame_by_name(
                                    selected_frame.replace(" ", "_")
                                )

                                random_frame_name = (
                                    random_parent_frame.name.replace("_", " ")
                                )

                                definition_transl = get_voted_translation(
                                    frame_name=random_frame_name,
                                    input_name=random_frame_name
                                )

                                if definition_transl:
                                    frame_def_fr = get_first_sentence(definition_transl)

                                    parents_text = (
                                        f"{random_frame_name} defined as {frame_def_fr}"
                                    )

                            except Exception:
                                parents_text = f"{random_frame_name}: Definition is not available"

                    for question in questions:

                        q_template = question["text"]
                        q_id = question["id"]

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame=random_frame_name,
                            definition_name=parents_text
                        )

                        print(question_text)

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "No",
                            "ids": "FR2A2"
                        })


                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "Uses",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else:
                    print(f"Frame '{frame_name}' does not have any Uses relations.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")


            print("\nQuestion 6.9:")
            question_samples = []
            question_samples_none = []


            # --- Config ---
            frame_name = frame_nam

            # --- Collect true parents via Inheritance (de-duplicated, and ensure sub = target) ---
            parents = []
            seen = set()
            f = fn.frame(frame_name.replace(" ", "_"))
            for fr in f["frameRelations"]:
                rel_type = fr["type"]["name"]
                parent = fr["superFrameName"]
                child = fr["subFrameName"]

                if rel_type == "Using" and child == frame_name.replace(" ", "_"):  # Arrest is the child
                    parents.append(parent)

            # --- Build MCQs (3 options): correct parent + one random distractor + "None of them" ---
            question_samples = []

            if parents:
                # pick ONE correct parent at random (if there are multiple)
                correct_parent = random.choice(parents)
                correct_p = correct_parent.replace("_", " ")
                definition_transl = get_voted_translation(frame_name=correct_p, input_name = correct_p)
                first_sentence = get_first_sentence(definition_transl)
                correct_option_text = f"{correct_p} — {first_sentence}"

                # pick ONE incorrect frame (not the target and not any parent)
                all_names = [f.name for f in all_frames if f.name != frame.name]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,2)
                incorrect_option_text = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_option_text.append(f"{frame_n} — {first_sentence}")

                # loop over your templates (expects template['frame_inheritance_prompts_mcq'])
                for q in template['uses_frame_prompts_mcq']:
                    options = [correct_option_text, *incorrect_option_text, "None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == correct_option_text)
                    matched_answer = f"{correct_letter}) {option_letters[correct_letter]}"
                    matched_answe = matched_answer.replace("_", " ")
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answe,
                        "ids": "PF3A1"
                    })


                # No parents → "None of them" is correct; use two random incorrect frames
                all_names = [f.name for f in all_frames if f.name != frame.name]
                incorrectt = random.sample(all_names,3)
                incorrect_options = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_options.append(f"{frame_n} — {first_sentence}")

                for q in template['uses_frame_prompts_mcq']:
                    options = incorrect_options + ["None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == "None of Them")
                    matched_answe = f"{correct_letter}) None of Them"
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answe,
                        "ids": "PF3A2"
                    })

            for selected_sample in question_samples:
              # Print the question and options
              print(f"{selected_sample['id']}: {selected_sample['question']}")
              for letter, option in selected_sample['options'].items():
                  print(f"{letter}) {option}")

              print(f"{selected_sample['ids']}: {selected_sample['answer']}")
              print("\n" + "-"*60 + "\n")

              # Create a string for the question and options
              options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
              combined_question = f"{selected_sample['question']}\n{options_string}"
              # Append the selected question and answer details to csv_data
              csv_data.append([
                  frame_name,
                  "Uses",
                  "3",
                  selected_sample['id'],
                  "MCQ",
                  selected_sample['ids'],
                  combined_question,
                  selected_sample['answer']
              ])

            else:
                print(f"Frame '{frame_name}' does not have any Uses relations.")



            print("ID: Samples")

            print("\nQuestion 6.10:")

            perspective_examples = []
            try:
                # Load the frame by name
                perspectives = []
                f = fn.frame(frame_nam.replace(" ", "_"))
                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]
                    parent = fr["superFrameName"]
                    child = fr["subFrameName"]

                    if rel_type == "Subframe" and child == frame_name.replace(" ", "_"):  #
                        perspectives.append(parent)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: \"{first_sentence}\"")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["subframe_open"]),
                                    len(template["subframe_open_ans"]))

                    for i in range(num_pairs):
                        q_template = template["subframe_open"][i]['text']
                        a_template = template["subframe_open_ans"][i]['text']
                        q_id = template["subframe_open"][i]['id']
                        a_id = template["subframe_open_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "SubFrame of",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else:
                    print(f"No 'Subframe' relations found for frame '{frame_name}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")




            print("\nQuestion 6.11:")
            perspective_on_binary = []
            perspectives = []
            try:
                # Use a consistent variable name
                f = fn.frame(frame_nam.replace(" ", "_"))
                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]
                    parent = fr["superFrameName"]
                    child = fr["subFrameName"]

                    if rel_type == "Subframe" and child == frame_name.replace(" ", "_"):  # Arrest is the child
                        perspectives.append(parent)

                if perspectives:
                    perspective_defs = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: {first_sentence}")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["subframe_open_binary"])

                    for i in range(num_pairs):
                        q_template = template["subframe_open_binary"][i]['text']
                        q_id = template["subframe_open_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No Logic

                    questions = template["subframe_open_binary"]

                    # Remove current frame
                    other_frames = [
                        f.name
                        for f in all_frames
                        if f.name != frame_name
                    ]

                    random.shuffle(other_frames)

                    # SAFE group size
                    group_size = max(1, len(perspectives))

                    target_groups = 20
                    needed_size = target_groups * group_size

                    other_frames = other_frames[:needed_size]

                    # Group candidates
                    frame_groups = [
                        ", ".join(other_frames[j:j + group_size])
                        for j in range(0, len(other_frames), group_size)
                    ]

                    while len(frame_groups) < target_groups:
                        frame_groups.append("")

                    # Perspective string
                    parent_frame_string = ", ".join(perspectives)

                    # Similarity
                    rand_group, score = get_first_low_similarity_candidate(
                        parent_frame_string.replace("_", " "),
                        frame_groups,
                        threshold=0.5
                    )

                    # Default values
                    random_frame_name = "None"
                    parents_text = "Definition is not available"

                    # Recover one frame
                    if rand_group and isinstance(rand_group, str):

                        candidate_frames = [
                            x.strip()
                            for x in rand_group.split(",")
                            if x.strip()
                        ]

                        selected_frame = (
                            random.choice(candidate_frames)
                            if candidate_frames else None
                        )

                        if selected_frame:
                            try:
                                random_parent_frame = fn.frame_by_name(
                                    selected_frame.replace(" ", "_")
                                )

                                random_frame_name = (
                                    random_parent_frame.name.replace("_", " ")
                                )

                                definition_transl = get_voted_translation(
                                    frame_name=random_frame_name,
                                    input_name=random_frame_name
                                )

                                if definition_transl:
                                    frame_def_fr = get_first_sentence(definition_transl)

                                    parents_text = (
                                        f"{random_frame_name} defined as  \"{frame_def_fr}\""
                                    )

                            except Exception:
                                parents_text = f"{random_frame_name}: Definition is not available"

                    for question in questions:

                        q_template = question["text"]
                        q_id = question["id"]

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame=random_frame_name,
                            definition_name=parents_text
                        )

                        print(question_text)

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "No",
                            "ids": "FR2A2"
                        })




                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "SubFrame of",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else:
                    print(f"No 'Subframe' relations found for frame '{frame_nam}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")


            print("\nQuestion 6.12:")

            perspective_examples = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Precedes":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: {first_sentence}")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["precedes_ordering"]),
                                    len(template["precedes_ordering_ans"]))

                    for i in range(num_pairs):
                        q_template = template["precedes_ordering"][i]['text']
                        a_template = template["precedes_ordering_ans"][i]['text']
                        q_id = template["precedes_ordering"][i]['id']
                        a_id = template["precedes_ordering_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Precedes",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Precedes' relations found for frame '{frame_name}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")




            print("\nQuestion 6.13:")
            perspective_on_binary = []
            perspectives = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Precedes":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)


                if perspectives:
                    perspective_defs = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: \"{first_sentence}\"")

                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["precedes_ordering_binary"])

                    for i in range(num_pairs):
                        q_template = template["precedes_ordering_binary"][i]['text']
                        q_id = template["precedes_ordering_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No Logic

                    questions = template["precedes_ordering_binary"]

                    if questions:
                  # Randomly pick  frames


                      parent_definitions = []
                      frame_names = []


                      # Remove current frame
                      other_frames = [f.name for f in all_frames if f.name != frame_name]

                      random.shuffle(other_frames)

                      target_groups = 20
                      group_size = len(perspectives)  # saf

                      needed_size = target_groups * group_size
                      other_frames = other_frames[:needed_size]

                      frame_groups = [
                          ", ".join(other_frames[i:i + group_size])
                          for i in range(0, len(other_frames), group_size)
                      ]

                      while len(frame_groups) < target_groups:
                          frame_groups.append("")

                      parent_frame_string = ", ".join(perspectives)

                      # Similarity on grouped candidates
                      rand_group, score = get_first_low_similarity_candidate(
                          parent_frame_string.replace("_", " "),
                          frame_groups,
                          threshold=0.5
                      )

                      random_frame_name = "None"
                      parents_text = "Definition is not available"

                      if rand_group:

                          if isinstance(rand_group, list) and rand_group:
                              selected_frame = random.choice(rand_group)

                          elif isinstance(rand_group, str):
                              selected_frame = random.choice(
                                  [x.strip() for x in rand_group.split(",") if x.strip()]
                              )

                          else:
                              selected_frame = None

                          if selected_frame:

                              try:
                                  random_parent_frame = fn.frame_by_name(selected_frame)

                                  random_frame_name = random_parent_frame.name.replace("_", " ")

                                  definition_transl = get_voted_translation(
                                      frame_name=random_frame_name,
                                      input_name=random_frame_name
                                  )

                                  if definition_transl:
                                      frame_def_fr = get_first_sentence(definition_transl)
                                      parents_text = f"{random_frame_name} defined as: \"{frame_def_fr}\""
                                  else:
                                      parents_text = f"{random_frame_name}: Definition is not available"

                              except Exception:
                                  parents_text = f"{random_frame_name}: Definition is not available"


                      for question in questions:

                          q_template = question["text"]
                          q_id = question["id"]

                          question_text = q_template.format(
                              frame_name=frame_nam.replace("_", " "),
                              parent_frame=random_frame_name,
                              definition_name=parents_text
                          )

                          print(question_text)

                          perspective_on_binary.append({
                              "question": question_text,
                              "id": q_id,
                              "answer": "No",
                              "ids": "FR2A2"
                          })



                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "Precedes",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Precedes' relations found for frame '{frame_nam}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")


            print("\nQuestion 6.14:")

            perspective_examples = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Inchoative_of":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: \"{first_sentence}\" ")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["Is_Inchoative_prompts"]),
                                    len(template["Is_Inchoative_prompts_ans"]))

                    for i in range(num_pairs):
                        q_template = template["Is_Inchoative_prompts"][i]['text']
                        a_template = template["Is_Inchoative_prompts_ans"][i]['text']
                        q_id = template["Is_Inchoative_prompts"][i]['id']
                        a_id = template["Is_Inchoative_prompts_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Is Inchoative of",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Inchoative_of' relations found for frame '{frame_name}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")




            print("\nQuestion 6.15:")
            perspective_on_binary = []
            perspectives = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Inchoative_of":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)


                if perspectives:
                    perspective_defs = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: \"{first_sentence}\" ")

                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["Is_Inchoative_prompts_binary"])

                    for i in range(num_pairs):
                        q_template = template["Is_Inchoative_prompts_binary"][i]['text']
                        q_id = template["Is_Inchoative_prompts_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No Logic

                    questions = template["Is_Inchoative_prompts_binary"]

                    # Remove current frame
                    other_frames = [
                        f.name
                        for f in all_frames
                        if f.name != frame_name
                    ]

                    random.shuffle(other_frames)

                    # SAFE group size
                    group_size = max(1, len(perspectives))

                    target_groups = 20
                    needed_size = target_groups * group_size

                    other_frames = other_frames[:needed_size]

                    # Group candidates
                    frame_groups = [
                        ", ".join(other_frames[j:j + group_size])
                        for j in range(0, len(other_frames), group_size)
                    ]

                    while len(frame_groups) < target_groups:
                        frame_groups.append("")

                    # Perspective string
                    parent_frame_string = ", ".join(perspectives)

                    # Similarity
                    rand_group, score = get_first_low_similarity_candidate(
                        parent_frame_string.replace("_", " "),
                        frame_groups,
                        threshold=0.5
                    )

                    # Default values
                    random_frame_name = "None"
                    parents_text = "Definition is not available"

                    # Recover one frame
                    if rand_group and isinstance(rand_group, str):

                        candidate_frames = [
                            x.strip()
                            for x in rand_group.split(",")
                            if x.strip()
                        ]

                        selected_frame = (
                            random.choice(candidate_frames)
                            if candidate_frames else None
                        )

                        if selected_frame:
                            try:
                                random_parent_frame = fn.frame_by_name(
                                    selected_frame.replace(" ", "_")
                                )

                                random_frame_name = (
                                    random_parent_frame.name.replace("_", " ")
                                )

                                definition_transl = get_voted_translation(
                                    frame_name=random_frame_name,
                                    input_name=random_frame_name
                                )

                                if definition_transl:
                                    frame_def_fr = get_first_sentence(definition_transl)

                                    parents_text = (
                                        f"{random_frame_name} defined as: \"{frame_def_fr}\" "
                                    )

                            except Exception:
                                parents_text = f"{random_frame_name}: Definition is not available"

                    for question in questions:

                        q_template = question["text"]
                        q_id = question["id"]

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame=random_frame_name,
                            definition_name=parents_text
                        )

                        print(question_text)

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "No",
                            "ids": "FR2A2"
                        })




                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "Is Inchoative of",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Inchoative_of' relations found for frame '{frame_nam}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")



            print("\nQuestion 6.16:")

            perspective_examples = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Causative_of":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)

                if perspectives:
                    perspective_defs = []
                    perspective_frame_names = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as: \"{first_sentence}\" ")
                            perspective_frame_names.append(other)
                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = min(len(template["Is_Causative_prompts"]),
                                    len(template["Is_Causative_prompts_ans"]))

                    for i in range(num_pairs):
                        q_template = template["Is_Causative_prompts"][i]['text']
                        a_template = template["Is_Causative_prompts_ans"][i]['text']
                        q_id = template["Is_Causative_prompts"][i]['id']
                        a_id = template["Is_Causative_prompts_ans"][i]['id']

                        question_text = q_template.format(frame_name=frame_name.replace("_", " "))

                        answer_text = a_template.format(
                            frame_name=frame_name.replace("_", " "),
                            parent_frame = other.replace("_", " "),
                            definition_name=perspective_text.replace("_", " ")
                        )

                        perspective_examples.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": answer_text,
                            "ids": a_id
                        })

                # Print and save to CSV buffer
                for pairs in perspective_examples:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_name,
                        "Is Causative of",
                        "1",
                        pairs['id'],
                        "Opened",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Causative_of' relations found for frame '{frame_name}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")




            print("\nQuestion 6.17:")
            perspective_on_binary = []
            perspectives = []
            try:
                frame_name = frame_nam
                f = fn.frame(frame_name.replace(" ", "_"))

                perspectives = []
                following = []

                for fr in f["frameRelations"]:
                    rel_type = fr["type"]["name"]

                    if rel_type == "Causative_of":
                        child = fr.get("subFrameName")
                        parent = fr.get("superFrameName")

                        # Case 1: Arrest precedes another frame
                        if parent == frame_name.replace(" ", "_"):
                            perspectives.append(child)


                if perspectives:
                    perspective_defs = []
                    for other in perspectives:
                        try:
                            definition_transl = get_voted_translation(frame_name=other.replace("_", " "), input_name = other.replace("_", " "))
                            first_sentence = get_first_sentence(definition_transl)
                            perspective_defs.append(f"{other} defined as \"{first_sentence}\" ")

                        except Exception:
                            perspective_defs.append(f"{other} Definition is not available")

                    # Format definitions
                    if len(perspective_defs) > 1:
                        perspective_text = ", ".join(perspective_defs[:-1]) + " and " + perspective_defs[-1]
                    else:
                        perspective_text = perspective_defs[0]

                    # Generate Q/A pairs
                    num_pairs = len(template["Is_Causative_prompts_binary"])

                    for i in range(num_pairs):
                        q_template = template["Is_Causative_prompts_binary"][i]['text']
                        q_id = template["Is_Causative_prompts_binary"][i]['id']

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame = perspective_text.replace("_", " "),
                            definition_name=other.replace("_", " ")
                        )

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "Yes",
                            "ids": "FPO2A1"
                        })


                                    #No samples


                    questions = template["Is_Causative_prompts_binary"]

                    # Remove current frame
                    other_frames = [
                        f.name
                        for f in all_frames
                        if f.name != frame_name
                    ]

                    random.shuffle(other_frames)

                    # SAFE group size
                    group_size = max(1, len(perspectives))

                    target_groups = 20
                    needed_size = target_groups * group_size

                    other_frames = other_frames[:needed_size]

                    # Group candidates
                    frame_groups = [
                        ", ".join(other_frames[j:j + group_size])
                        for j in range(0, len(other_frames), group_size)
                    ]

                    while len(frame_groups) < target_groups:
                        frame_groups.append("")

                    # Perspective string
                    parent_frame_string = ", ".join(perspectives)

                    # Similarity
                    rand_group, score = get_first_low_similarity_candidate(
                        parent_frame_string.replace("_", " "),
                        frame_groups,
                        threshold=0.5
                    )

                    # Default values
                    random_frame_name = "None"
                    parents_text = "Definition is not available"

                    # Recover one frame
                    if rand_group and isinstance(rand_group, str):

                        candidate_frames = [
                            x.strip()
                            for x in rand_group.split(",")
                            if x.strip()
                        ]

                        selected_frame = (
                            random.choice(candidate_frames)
                            if candidate_frames else None
                        )

                        if selected_frame:
                            try:
                                random_parent_frame = fn.frame_by_name(
                                    selected_frame.replace(" ", "_")
                                )

                                random_frame_name = (
                                    random_parent_frame.name.replace("_", " ")
                                )

                                definition_transl = get_voted_translation(
                                    frame_name=random_frame_name,
                                    input_name=random_frame_name
                                )

                                if definition_transl:
                                    frame_def_fr = get_first_sentence(definition_transl)

                                    parents_text = (
                                        f"{random_frame_name} defined as {frame_def_fr}"
                                    )

                            except Exception:
                                parents_text = f"{random_frame_name}: Definition is not available"

                    for question in questions:

                        q_template = question["text"]
                        q_id = question["id"]

                        question_text = q_template.format(
                            frame_name=frame_nam.replace("_", " "),
                            parent_frame=random_frame_name,
                            definition_name=parents_text
                        )

                        print(question_text)

                        perspective_on_binary.append({
                            "question": question_text,
                            "id": q_id,
                            "answer": "No",
                            "ids": "FR2A2"
                        })


                # Print results
                for pairs in perspective_on_binary:
                    print(f"{pairs['id']}: {pairs['question']}")
                    print(f"{pairs['ids']}: {pairs['answer']}")
                    print("\n" + "-"*60 + "\n")

                    csv_data.append([
                        frame_nam,
                        "Is Causative of",
                        "2",
                        pairs['id'],
                        "Closed",
                        pairs['ids'],
                        pairs['question'],
                        pairs['answer']
                    ])
                else: print(f"No 'Causative_of' relations found for frame '{frame_nam}'.")

            except Exception as e:
                print(f"Frame '{frame_name}' not available or error occurred: {str(e)}")



            print("\nQuestion 6.18:")
            question_samples = []
            question_samples_none = []


            # --- Config ---
            frame_name = frame_nam
            f = fn.frame(frame_name.replace(" ", "_"))
            frame = fn.frame_by_name(frame_name.replace(" ", "_"))

            parents = []
            following = []

            for fr in f["frameRelations"]:
                rel_type = fr["type"]["name"]

                if rel_type == "Causative_of":
                    child = fr.get("subFrameName")
                    parent = fr.get("superFrameName")

                    # Case 1: Arrest precedes another frame
                    if parent == frame_name.replace(" ", "_"):
                        parents.append(child)


            # --- Build MCQs (3 options): correct parent + one random distractor + "None of them" ---

            if parents:
                # pick ONE correct parent at random (if there are multiple)
                correct_parent = random.choice(parents)
                correct_p = correct_parent.replace("_", " ")
                definition_transl = get_voted_translation(frame_name=correct_p, input_name = correct_p)
                first_sentence = get_first_sentence(definition_transl)
                correct_option_text = f"{correct_p} — {first_sentence}"


                all_names = [f.name for f in all_frames if f.name != frame.name]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,2)
                incorrect_option_text = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_option_text.append(f"{frame_n} — {first_sentence}")

                # loop over your templates (expects template['frame_inheritance_prompts_mcq'])
                for q in template['Is_Causative_prompts_mcq']:
                    options = [correct_option_text, *incorrect_option_text, "None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == correct_option_text)
                    matched_answer = f"{correct_letter}) {option_letters[correct_letter]}"
                    matched_answe = matched_answer.replace("_", " ")
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answe,
                        "ids": "PF3A1"
                    })


                # No parents → "None of them" is correct; use two random incorrect frames
                all_names = [f.name for f in all_frames if f.name != frame.name]
                #distractor_pool = [n for n in all_names if n not in set(parents + [frame.name])]
                incorrectt = random.sample(all_names,3)
                incorrect_options = []
                for incorrect in incorrectt:
                    frame_n = incorrect
                    definition_transl = get_voted_translation(frame_name=frame_n.replace("_", " "), input_name = frame_n.replace("_", " "))
                    first_sentence = get_first_sentence(definition_transl)
                    incorrect_options.append(f"{frame_n} — {first_sentence}")

                for q in template['Is_Causative_prompts_mcq']:
                    options = incorrect_options + ["None of Them"]
                    random.shuffle(options)
                    option_letters = {chr(97+i): options[i] for i in range(len(options))}

                    #print(f"{q['id']}: {q['text'].format(frame_name=frame_name)}")
                    #for k, v in option_letters.items():
                        #print(f"{k}) {v}")

                    correct_letter = next(k for k, v in option_letters.items() if v == "None of Them")
                    matched_answe = f"{correct_letter}) None of Them"
                    #print(f"Answer: {matched_answer}\n")

                    question_samples.append({
                        "question": q['text'].format(frame_name=frame_name.replace("_", " ")),
                        "id": q['id'],
                        "options": option_letters,
                        "answer": matched_answe,
                        "ids": "PF3A2"
                    })

            for selected_sample in question_samples:
              # Print the question and options
              print(f"{selected_sample['id']}: {selected_sample['question']}")
              for letter, option in selected_sample['options'].items():
                  print(f"{letter}) {option}")

              print(f"{selected_sample['ids']}: {selected_sample['answer']}")
              print("\n" + "-"*60 + "\n")

              # Create a string for the question and options
              options_string = "".join(f"{letter}) {option}\n" for letter, option in selected_sample['options'].items())
              combined_question = f"{selected_sample['question']}\n{options_string}"
              # Append the selected question and answer details to csv_data
              csv_data.append([
                  frame_name,
                  "Is Causative of",
                  "3",
                  selected_sample['id'],
                  "MCQ",
                  selected_sample['ids'],
                  combined_question,
                  selected_sample['answer']
              ])
            else: print(f"No 'Causative_of' relations found for frame '{frame_nam}'.")


            print("\n" + "="*50 + "\n")

            return frame_nam, csv_data

        # Function to write the data to a CSV file
        def write_to_csv(frame_nam, csv_data):
            if not csv_data:
                print(f"No data to write for frame '{frame_nam}'")
                return

            # Ensure directory exists
            os.makedirs(target_folder, exist_ok=True)

            frame_name_clean = frame_nam.replace("_", " ")
            file_name = os.path.join(target_folder, f"{frame_name_clean}.csv")

            try:
                with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)

                    writer.writerow([
                        "Frame Name",
                        "Prompt Type",
                        "Question ID",
                        "Question Sample ID",
                        "Answer Type",
                        "Answer Sample ID",
                        "Questions",
                        "Answers"
                    ])

                    writer.writerows(csv_data)

                print(f"Data written successfully to: {file_name}")

            except Exception as e:
                print(f"Error writing CSV: {e}")


        # Main function
        def main():
            global frame_name, VARIANT

            if not frame_name:
                print("No valid frame name returned")
                return

            frame_data = {
                "name": frame_name,
            }

            try:
                VARIANT = 0
                result = fill_template(frame_data)

                if not result:
                    print("Skipping: fill_template returned None")
                    return

                frame_name_result, csv_data = result

                VARIANT = 1
                _, second_pass = fill_template(frame_data)
                csv_data = merge_passes(csv_data, second_pass)

                print("Returned frame:", frame_name_result)
                print("CSV rows:", len(csv_data))

                write_to_csv(frame_name_result, csv_data)

            except Exception as e:
                print(f"Unexpected error in main(): {e}")


        # Run script
        if __name__ == "__main__":
            main()