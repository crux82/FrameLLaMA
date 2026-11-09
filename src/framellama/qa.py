"""Question type of a FrameNet instruction example.

The released files have no answer-type field, so the type is read from the
answer, as written by the generator: "Yes"/"No" for closed questions, an
option label such as "c) ..." for multiple choice, free text otherwise.
On generator output this rule agrees with the generator's own answer_type
label for every record.
"""

import re

TYPES = {"closed": "Closed", "mcq": "MCQ", "opened": "Opened"}


def answer_type(answer):
    a = str(answer or "").strip()
    if a in ("Yes", "No"):
        return "closed"
    if re.match(r"^[a-eA-E]\)", a):
        return "mcq"
    return "opened"
