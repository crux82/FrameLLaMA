"""SRL prompts and output parsing, copied verbatim from the original scripts.

Generated from Evaluations/SRL/eval_srl_vlm_{roles,lu,frame}.py. Only the
function names were changed (build_prompt -> build_prompt_<task>, and the
two different filter_json_by_frame versions are kept apart).
"""

import json
import re




# ---- from eval_srl_vlm_roles.py ----



def build_prompt_roles(frame_combine, lexical_combine, frame, sentence, lexical_unit):
    return f"""
You are an SRL system trained on FrameNet.
Your task is to extract the frame elements expressed by the lexical unit.
and return their exact spans from the sentence.

FOCUS ON TWO REQUIREMENTS:

A) FRAME ELEMENT IDENTIFICATION
   - Extract ONLY the frame elements that belong to the given frame.
   - A frame element(FE) must be explicitly expressed in the text.
   - Ignore any FE that is not evoked by this lexical unit in this sentence.
   - Do NOT invent or hallucinate FE.

B) PRECISE SPAN MAPPING
   - For every FE you extract, return the exact substring from the sentence.
   - Do NOT paraphrase, shorten, or expand spans.
   - The FE span must come from the same clause or phrase where the lexical unit appears.

OUTPUT CONSTRAINT (STRICT):
- You MUST output a single valid JSON object.
- You MUST NOT output any text before '{' and after with '}' even single letter is not allowed.
- You MUST NOT repeat the JSON.
- You MUST NOT include explanations, comments, examples, or markdown.

READ-ONLY CONTEXT (DO NOT OUTPUT)
{{
  "input_sentence": "{sentence}",
  "frames_in_sentence": "{frame_combine}",
  "lexical_units_in_sentence": "{lexical_combine}"
}}

PURPOSE OF THIS CONTEXT:
The sentence may evoke multiple frames in different clauses or phrases.
Use this context only to locate the correct evoked region for the TARGET
lexical unit and avoid extracting roles from other events.

NOW PROCESS THE REAL JSON INPUT:
{{
  "input_sentence": "{sentence}",
  "target_frame": "{frame}",
  "target_lexical_unit": "{lexical_unit}"
}}

COMPLETE THE JSON OBJECT AS PER THE FORMAT BELOW:
{{
  "input_sentence": "{sentence}",
  "annotations": [
    {{
      "frame": "{frame}",
      "lexical_unit": "{lexical_unit}",
      "frame_elements": {{
        "<FrameElement>": "<Exact span from the sentence>"
        "<FrameElement>": "<Exact span from the sentence>"

      }}
    }}
  ]
}}"""


def safe_parse_json(raw_text):
    """
    Safely extract and parse a JSON object from malformed LLM output.
    If parsing fails, return the original raw_text.
    """

    if raw_text is None:
        return raw_text

    text = raw_text.strip()

    # --------------------------------------------------
    # STEP 1: If text is quoted, UNQUOTE IT FIRST
    # --------------------------------------------------
    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except Exception:
            # cannot unwrap → keep going with original
            text = text[1:-1]

    # --------------------------------------------------
    # STEP 2: Remove everything BEFORE first '{'
    # --------------------------------------------------
    first_brace = text.find("{")
    if first_brace == -1:
        return raw_text  # no JSON possible

    text = text[first_brace:].strip()

    # --------------------------------------------------
    # STEP 3: Extract EXACT balanced JSON block
    # --------------------------------------------------
    stack = []
    buf = ""

    for ch in text:
        if ch == "{":
            stack.append("{")
        if stack:
            buf += ch
        if ch == "}":
            stack.pop()
            if not stack:
                break

    # --------------------------------------------------
    # STEP 4: Final parse attempt
    # --------------------------------------------------
    try:
        return json.loads(buf)
    except Exception:
        return raw_text



# ---- from eval_srl_vlm_lu.py ----



def build_prompt_lu(frame, sentence, lexical_unit):
    return f"""
You are trained on FrameNet.
Your task is to extract ONLY the lexical unit that are expressed
in the part of the sentence where the given frame evokes.

RULES:
1. Identify the phrase or clause in which the frame appears.
2. Extract only the lexical unit present in this frame-evoked region.
3. Do NOT extract lexical unit from other parts of the sentence.
4. Do NOT invent lexical unit. Use only those belonging to the given frame.
5. Output strictly valid JSON.

Expected OUTPUT:
{{
  "input_sentence": "Hall, who recently returned from a trip to Iraq, said sanctions hurt civilians.",
  "annotations": [
          {{
			  "frame": "Location",
			  "lexical unit": "trip"
		   }}
      ]
}}

NOW PROCESS THE REAL INPUT:

FRAME: {frame}
SENTENCE: {sentence}

REQUIRED JSON FORMAT:
{{
  "input_sentence": "{sentence}",
  "annotations": [
          {{
			  "frame":"{frame}"
			   "lexical_unit": "<Exact lexical unit from frame-evoked part>"
		  }}
      ]
}}
        
        """


def extract_json_blocks(text):
    json_blocks, stack, buf = [], [], ""
    inside = False

    for c in text:
        if c == "{":
            inside = True
            stack.append("{")
            buf += c
        elif c == "}":
            if stack:
                stack.pop()
                buf += c
                if not stack:
                    json_blocks.append(buf)
                    buf, inside = "", False
        elif inside:
            buf += c
    return json_blocks


def remove_newlines_from_output(raw_output):
    cleaned = []
    for item in raw_output:
        if isinstance(item, list) and len(item) == 1:
            item = item[0]
        if isinstance(item, str):
            cleaned_item = item.replace("\\n", " ").replace("\n", " ").replace("\\t", " ").strip()
            cleaned.append(cleaned_item)
    return cleaned


def sanitize_json_text(text):
    text = text.replace("\\n", " ").replace("\n", " ").replace("\\t", " ").replace('"{', '[{').replace('}"', '}]').strip()
    text = re.sub(r'\[" "]', '', text)
    text = re.sub(r'"([A-Za-z0-9_\- ]+):[^"]*"\s*:', lambda m: f'"{m.group(1).strip()}":', text)
    text = re.sub(r'"([A-Za-z0-9_\- ]+): ([^"]+)"', lambda m: f'"{m.group(1).strip()}": "{m.group(2).strip()}"', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def filter_json_blocks_by_frame(json_blocks, target_frame):
    filtered = []
    seen = set()
    for block in json_blocks:
        try:
            cleaned = sanitize_json_text(block)
            parsed = json.loads(cleaned)
            for ann in parsed.get("annotations", []):
                frame = ann.get("frame")
                if frame == target_frame and frame not in seen:
                    filtered.append(parsed)
                    seen.add(frame)
                    break
        except json.JSONDecodeError:
            continue
    return filtered



# ---- from eval_srl_vlm_frame.py ----



def build_prompt_frame(frame, sentence, lexical_unit):
    return f"""
You are an expert in Frame Semantics and FrameNet.
Your task is to determine which FrameNet frame is evoked by the given lexical unit in the sentence.

INSTRUCTIONS:
1. Analyze the provided sentence.
2. Focus ONLY on the given lexical unit as the target of analysis.
3. Identify the FrameNet frame that this lexical unit evokes in this specific sentence.
4. Do NOT guess frames unrelated to the lexical unit.
5. Do NOT output definitions, explanations, or extra text.
6. The frame MUST be a valid FrameNet frame name.
7. Output strictly valid JSON.

EXAMPLE
Lexical Unit: trip
Sentence: Hall recently returned from a trip to Iraq.

Expected Output:
{{
  "input_sentence": "Hall recently returned from a trip to Iraq.",
  "annotations": [
    {{
      "lexical_unit": "trip",
      "frame": "Travel"
    }}
  ]
}}

NOW PROCESS THE REAL INPUT:

LEXICAL UNIT: {lexical_unit}
SENTENCE: {sentence}

REQUIRED JSON FORMAT:
{{
  "input_sentence": "{sentence}",
  "annotations": [
    {{
      "lexical_unit": "{lexical_unit}",
      "frame": "<Predicted FrameNet frame name>"
    }}
  ]
}}"""
