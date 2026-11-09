"""Generation of the FrameNet instruction-tuning examples (40 task families).

The body of this module is copied verbatim (indentation reduced by 8 spaces)
from the original research script framenet_dataset_generator.py, lines
865-5991, by tools/split_original_generator.py. Do not edit it by hand.

In the original script this code ran inside a module-level loop over frames
and read a few module-level names (fn, all_frames, frame_name, template, ...).
setup() provides the same names here, and generate_frame() runs the original
two-pass procedure for one frame:

    VARIANT = 0 -> fill_template()   (first template variant)
    VARIANT = 1 -> fill_template()   (second variant, fresh distractors)
    merge_passes()                   (keep pass 2 only for PARAPHRASED_TASKS)

Source SHA-256: 3efa95ff772c9ed5d4fe5d79a4f30762adf126c01c25fb5a05fa5b5c6b40f8c7
"""

import csv
import os
import random
import re

from . import framenet_reader as _reader
from . import negatives as _negatives
from .framenet_reader import (
    get_first_sentence,
    get_urdu_lexical_units,
    get_urdu_lexical_units_def,
    get_voted_translated_examples,
    get_voted_translation,
    norm,
    related_frames,
    to_urdu_pos,
    valid_lexical_units,
)
from .negatives import get_first_low_similarity_candidate, negative_frames
from .templates import PARAPHRASED_TASKS, TEMPLATE

# ---------------------------------------------------------------
# Names the original code expects at module level (set by setup())
# ---------------------------------------------------------------
fn = None              # FrameNet reader
all_frames = []        # every FrameNet frame, in FrameNet order
frame_name = None      # frame currently being generated
template = TEMPLATE
target_folder = None   # only used by write_to_csv()

# =========================================================
# SAMPLING STRATEGY (T1-T40)
# =========================================================
VARIANT = 0


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


# =================================================================
# ORIGINAL GENERATION CODE (verbatim from the research script)
# =================================================================





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


# =================================================================
# ENTRY POINTS USED BY scripts/generate_dataset.py
# =================================================================
CSV_COLUMNS = [
    "Frame Name",
    "Prompt Type",
    "Question ID",
    "Question Sample ID",
    "Answer Type",
    "Answer Sample ID",
    "Questions",
    "Answers",
]


def setup(framenet_dir=None, download=False, embedding_model=_negatives.EMBEDDING_MODEL,
          negative_threshold=_negatives.NEGATIVE_THRESHOLD, csv_dir=None, show_progress=True):
    """Load FrameNet, the lookup tables and the negative-frame index."""
    global fn, all_frames, target_folder
    fn = _reader.load(framenet_dir, download=download)
    all_frames = list(fn.frames())
    _negatives.load_model(embedding_model)
    _negatives.build_negative_index(all_frames, threshold=negative_threshold,
                                    show_progress=show_progress)
    target_folder = csv_dir
    return fn


def generate_frame(name, seed=None):
    """Run the original two-pass generation for one frame.

    ``name`` is a FrameNet frame name (underscores or spaces). Returns the list
    of CSV rows (see CSV_COLUMNS). If the original code raises an error for
    this frame, the error is re-raised (the original script skipped the frame).
    """
    global frame_name, VARIANT

    frame_name = name.replace("_", " ")
    if seed is not None:
        random.seed(f"{seed}:{frame_name}")

    frame_data = {"name": frame_name}

    VARIANT = 0
    result = fill_template(frame_data)
    if not result or result[1] is None:
        return []
    _, csv_data = result

    VARIANT = 1
    _, second_pass = fill_template(frame_data)
    return merge_passes(csv_data, second_pass)
