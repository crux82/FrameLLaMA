"""The 40 task families: task key -> task number in the paper (T1-T40).

The task key is "<Prompt Type> | Q<Question ID>", as written by the
generator. Numbers and descriptions follow Table 7 (Appendix E) of the paper;
each key was matched by answer type and question template.

Note on T36-T40: the paper lists Causative as T36-T37 and Inchoative as
T38-T40, with T40 an MCQ. In the code the only MCQ of this group is generated
from the Is_Causative_of relation ("Is Causative of | Q3"), so T40 maps to it.
"""

TASK_FAMILIES = {
    "Frame Definition | Q1": "T1",          # Open: question about frame definition
    "Frame Definition | Q2": "T2",          # Open: detect frame from definition
    "Frame Definition | Q3": "T3",          # Yes/No: match/mismatch of frame definitions
    "Frame Elements | Q1": "T4",            # Open: FE inventory
    "Frame Elements | Q6": "T5",            # Open: definition of specific FE roles
    "Frame Elements | Q2": "T6",            # Open: detect frame from subset of FEs
    "Frame Elements | Q5": "T7",            # Yes/No: membership of FE in target frame
    "Frame Elements | Q7": "T8",            # Yes/No: validity of FE-specific definitions
    "Frame Elements | Q3": "T9",            # MCQ: contributor vs. non-contributor roles
    "Frame Elements | Q4": "T10",           # MCQ: identifying exclusive roles
    "Semantic Type | Q1": "T11",            # Open: semantic category of specific FEs
    "Semantic Type | Q2": "T12",            # Yes/No: FE-to-semantic-type mapping validity
    "Semantic Type | Q3": "T13",            # MCQ: selection of semantic type
    "Lexical Units | Q1": "T14",            # Open: LU triggers for target frame
    "Lexical Units | Q3": "T15",            # Open: functional use of LU in frame context
    "Lexical Units | Q2": "T16",            # Yes/No: LU-POS suitability for frame
    "Lexical Units | Q4": "T17",            # Yes/No: LU definition vs. frame evocation
    "Lexical Unit Polysemy | Q1": "T18",    # Open: LU polysemy to frame senses
    "Lexical Unit Polysemy | Q2": "T19",    # Yes/No: polysemous sense-to-frame linking
    "Lexical Units | Q5": "T20",            # MCQ: discriminating between frame senses
    "Frame Elements Examples | Q1": "T21",  # Open: example sentences for specific FEs
    "Frame Elements Examples | Q2": "T22",  # Yes/No: FE role binding in annotated examples
    "Inherit From | Q1": "T23",             # Open: inheritance, parent frame
    "Inherit From | Q2": "T24",             # Yes/No: inheritance link validity
    "Inherit from | Q3": "T25",             # MCQ: inheritance, child from parent (lower-case "from" as in the code)
    "Perspective On | Q1": "T26",           # Open: alternative viewpoint frame
    "Perspective On | Q2": "T27",           # Yes/No: viewpoint link verification
    "Perspective On | Q3": "T28",           # MCQ: discriminating viewpoints
    "Uses | Q1": "T29",                     # Open: structural foundation frame
    "Uses | Q2": "T30",                     # Yes/No: reused concept verification
    "Uses | Q3": "T31",                     # MCQ: base frame discrimination
    "SubFrame of | Q1": "T32",              # Open: component event identification
    "SubFrame of | Q2": "T33",              # Yes/No: part-of relation validity
    "Precedes | Q1": "T34",                 # Open: temporal sequence ordering
    "Precedes | Q2": "T35",                 # Yes/No: sequential link verification
    "Is Causative of | Q1": "T36",          # Open: causative, resulting state
    "Is Causative of | Q2": "T37",          # Yes/No: causative link validity
    "Is Inchoative of | Q1": "T38",         # Open: inchoative, resulting state
    "Is Inchoative of | Q2": "T39",         # Yes/No: inchoative outcome verification
    "Is Causative of | Q3": "T40",          # MCQ: discriminating outcomes (see note above)
}
