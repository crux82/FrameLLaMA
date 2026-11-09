"""SRL scoring: lexical units, frame elements, roles, spans, partial matches.

Original script: Evaluation_SRL_F1_Score_Roles.py. The scoring code is
unchanged; it was wrapped in score_roles() so file paths can be passed in.

ARGUMENT ORDER (kept from the original script, which produced the paper's
numbers): the first argument reference receives the MODEL OUTPUT file
(*_result_*.json) and the second argument predictions receives the
aligned GOLD file (*_gold-std_*.json). Because of this, the values printed as
"Precision" and "Recall" correspond to recall and precision in the usual
sense (gold as reference). F1 from exact matching is symmetric and not
affected; the one-word and 25%-overlap matchers are not symmetric, so the
order must be kept to reproduce the published numbers.
"""
import json
import nltk
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from typing import List, Tuple

STOPWORDS = set(["a", "an", "the", "in", "on", "at", "and", "of", "to", "is", "that", "it", "for", "with", "as", "was", "were", "by", "this", "which", "or", "be", "are"])
def extract_frames_and_lus(annotations):

    frames_lus = defaultdict(list)  # Automatically handles missing keys
    frames = set()

    for ann in annotations:
        fram = ann.get("frame", "").strip().lower()
        frame = fram.replace("_", " ")
        lu = ann.get("lexical_unit", "").strip().lower()

        if frame and lu:  # Ensure both frame and LU are present
            frames_lus[frame].append(lu)
            frames.add(frame)  # Store unique frame names

    return dict(frames_lus), frames

def extract_frame_elements(annotations):
    """Extract frame elements grouped by frame."""
    frame_elements = defaultdict(list)
    roles = defaultdict(list)
    arguments = defaultdict(list)
    for ann in annotations:
        frame = ann.get("frame", "").strip().lower()
        if frame and "frame_elements" in ann:
            for role, argument in ann["frame_elements"].items():
                if argument.strip():
                    frame_elements[frame].append((role.strip().lower(), argument.strip().lower())) # tuple of (role, arguments)
                    roles[frame].append((role.strip().lower())) # tuple of (role)
                    arguments[frame].append(( argument.strip().lower())) # tuple of (arguments)
    return frame_elements, roles, arguments


def word_based_similarity(ref_text, pred_text):

    ref_words = set(ref_text.lower().split())  # Convert reference text to set of words
    pred_words = set(pred_text.lower().split())  # Convert predicted text to set of words

    if not ref_words or not pred_words:  # Handle empty cases
        return 0

    common_words = ref_words & pred_words  # Exact word matches only
    total_words = len(ref_words)  # Normalize by reference word count

    return (len(common_words) / total_words) * 100  # Percentage similarity


def fuzzy_match(ref_set, pred_set, threshold=25):

    matched = []  # Store matched pairs directly in a list to preserve duplicates
    unmatched_ref = ref_set.copy()  # Keep a list of unmatched reference items
    unmatched_pred = pred_set.copy()  # Keep a list of unmatched predicted items

    # Group predictions by role for role-specific matching
    pred_dict = defaultdict(list)
    for role, span in pred_set:
        pred_dict[role].append(span)

    # Iterate over each reference annotation and find the best match within the same role
    for ref_role, ref_arg in ref_set:
        best_match = None
        highest_similarity = 0

        if ref_role in pred_dict:  # Only compare with the same role
            for pred_arg in pred_dict[ref_role]:
                similarity = word_based_similarity(ref_arg, pred_arg)  # Word-based similarity

                # Only consider matches above the threshold
                if similarity >= threshold and similarity > highest_similarity:
                    best_match = pred_arg
                    highest_similarity = similarity

        if best_match:
            matched.append((ref_role, best_match))  # Store match as (role, pred_text)
            unmatched_ref.remove((ref_role, ref_arg))  # Remove matched reference
            pred_dict[ref_role].remove(best_match)  # Remove matched prediction

    # Convert remaining unmatched predictions to a flat list
    unmatched_pred = [(role, span) for role, spans in pred_dict.items() for span in spans]

    return matched, unmatched_ref, unmatched_pred

def one_word_found(ref_list: List[Tuple[str, str]], pred_list: List[Tuple[str, str]]):
    matched = []
    unmatched_ref = ref_list.copy()  # Avoid modifying original list while iterating
    unmatched_pred = pred_list.copy()
    match_details = []

    # Track matched frame elements to avoid double-counting
    matched_frame_elements = defaultdict(int)

    # Iterate through all the reference and prediction pairs
    for ref_role, ref_argument in ref_list:
        ref_words = set(ref_argument.lower().split())
        best_match = None
        best_matched_word = None

        for pred_role, pred_argument in pred_list:
            if ref_role != pred_role:
                continue  # Roles must match exactly

            # Skip matching if this specific reference role is already matched
            if matched_frame_elements[(ref_role, ref_argument)] > 0:
                continue

            pred_words = set(pred_argument.lower().split())
            common_words = ref_words & pred_words

            if common_words:
                best_match = (pred_role, pred_argument)
                best_matched_word = next(iter(common_words))  # Get the first matched word
                matched_frame_elements[(ref_role, ref_argument)] += 1
                break  # Stop searching after the first found match

        if best_match:
            matched.append((ref_role, best_match[1]))  # Store matched role & prediction text
            match_details.append((ref_role, ref_argument, best_match[1], best_matched_word))

            # Remove matched items safely
            unmatched_ref = [item for item in unmatched_ref if item != (ref_role, ref_argument)]
            unmatched_pred = [item for item in unmatched_pred if item != best_match]

    return matched, unmatched_ref, unmatched_pred, match_details



from collections import Counter

def compute_f1_score(true_list, pred_list):
    true_counts = Counter(true_list)
    pred_counts = Counter(pred_list)

    # Count true positives as the minimum occurrences in both
    tp = sum(min(true_counts[elem], pred_counts[elem]) for elem in pred_counts)

    # Precision: TP / (TP + FP)
    precision = tp / sum(pred_counts.values()) if pred_counts else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / sum(true_counts.values()) if true_counts else 0.0

    # F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1





def score_roles(reference, predictions):
    """Original scoring code (unchanged). See module docstring for argument order."""
    print(f"Reference Sentences: {len(reference)}")
    print(f"Predictions Sentences: {len(predictions)}")
    # Ensure both reference and predictions have the same length
    #assert len(reference) == len(predictions), "Reference and Predictions must have the same number of sentences."
    # Filtering step
    filtered_reference, filtered_predictions = [], []
    removed_sentences, removed_frames = 0, 0

    for ref, pred in zip(reference, predictions):
        ref_sentences = ref.get("input_sentence", [])  # Extract sentences
        pred_sentences = pred.get("input_sentence", [])  # Ensure predictions are aligned

        ref_annotations = ref.get("annotations", [])
        pred_annotations = pred.get("annotations", [])
        #print(f"{ref_sentences[:20]}: {pred_sentences[:20]}")

        # Extract frames, lexical units, and frame elements
        ref_lus, ref_frames = extract_frames_and_lus(ref_annotations)
        pred_lus, pred_frames = extract_frames_and_lus(pred_annotations)
        ref_fes = extract_frame_elements(ref_annotations)
        pred_fes = extract_frame_elements(pred_annotations)


            # Remove sentences where either reference or prediction has empty annotations or frames
        if not (ref_annotations and pred_annotations and ref_frames and pred_frames):
            removed_sentences += 1
            removed_frames += len(ref_frames) + len(pred_frames)  # Count removed frames from both
            continue  # Skip this sentence


        filtered_reference.append(ref)
        filtered_predictions.append(pred)

    #print(f"Removed Sentences: {removed_sentences}")
    #print(f"Removed Frames: {removed_frames}")
    #print(f"Remaining Reference Sentences: {len(filtered_reference)}")
    #print(f"Remaining Predictions Sentences: {len(filtered_predictions)}")

    # List to store sentence-level F1 scores

    sentence_f1_scores_lus = []
    sentence_f1_scores = []
    sentence_f1_scores_ow = []
    sentence_f1_scores_od = []
    sentence_f1_scores_roles = []
    sentence_f1_scores_span = []
    frame_f1_scores_sm = []
    sentence_f1_scores_sm = []

    # Process each input instance
    for ref, pred in zip(reference, predictions):
        ref_sentences = ref.get("input_sentence", [])  # Extract sentences
        pred_sentences = pred.get("input_sentence", [])  # Ensure predictions are aligned
        #print(f"{ref_sentences[:20]}: {pred_sentences[:20]}")
        ref_fes, ref_roles, ref_arguments = extract_frame_elements(ref.get("annotations", []))
        pred_fes, pred_roles, pred_arguments = extract_frame_elements(pred.get("annotations", []))
        ref_lus, ref_frames = extract_frames_and_lus(ref["annotations"])
        pred_lus, pred_frames = extract_frames_and_lus(pred["annotations"])
        #print(f"{ref_frames}: {pred_frames}")

        frame_f1_scores_lus = []
        frame_f1_scores = []  # Stores F1 scores of all frames in the current sentence
        frame_f1_scores_ow = []
        frame_f1_scores_od = []
        frame_f1_scores_roles = []
        frame_f1_scores_span = []


        # Compute F1 for each frame in lexical units
        for frame in set(ref_lus.keys()).union(set(pred_lus.keys())):
            ref_list = ref_lus.get(frame, [])
            pred_list = pred_lus.get(frame, [])
            precision, recall, f1 = compute_f1_score(ref_list, pred_list)
            frame_f1_scores_lus.append((frame, precision, recall, f1))
            #print(f"Frame: {f1}: {precision}: {recall}: {frame}")

        # Compute F1 for each frame in frame elements
        for frame in set(ref_fes.keys()).union(set(pred_fes.keys())):
            ref_list = ref_fes.get(frame, [])
            pred_list = pred_fes.get(frame, [])
            f1_score = compute_f1_score(ref_list, pred_list)
            frame_f1_scores.append(f1_score)
            #print(f"Frame: {f1_score}: {frame}: {ref_list}: {pred_list}")

        # Compute F1 for each frame in frame elements roles only
        for frame in set(ref_roles.keys()).union(set(pred_roles.keys())):
            ref_list = ref_roles.get(frame, [])
            pred_list = pred_roles.get(frame, [])
            precision, recall, f1 = compute_f1_score(ref_list, pred_list)
            frame_f1_scores_roles.append((precision, recall, f1))

        # Compute F1 for each frame in frame elements Spans only
        for frame in set(ref_arguments.keys()).union(set(pred_arguments.keys())):
            ref_list = ref_arguments.get(frame, [])
            pred_list = pred_arguments.get(frame, [])
            f1_score = compute_f1_score(ref_list, pred_list)
            frame_f1_scores_span.append(f1_score)

        # Compute F1 for each frame in frame elements one word matching
        for frame in set(ref_fes.keys()).union(set(pred_fes.keys())):
            ref_list, pred_list = ref_fes.get(frame, []), pred_fes.get(frame, [])
            # Apply one-word matching function
            matched, unmatched_ref, unmatched_pred, _ = one_word_found(ref_list, pred_list)
            # Use matched elements + unmatched ones for F1 calculation
            updated_ref_list = matched + unmatched_ref
            updated_pred_list = matched + unmatched_pred
            precision, recall, f1 = compute_f1_score(updated_ref_list, updated_pred_list)
            frame_f1_scores_ow.append((precision, recall, f1))

            # Compute F1 for each frame in frame elements on simalrity based
        for frame in set(ref_fes.keys()).union(set(pred_fes.keys())):
            ref_list = ref_fes.get(frame, [])
            pred_list = pred_fes.get(frame, [])

            threshold_value = 25  # Only exact word matches will count

            matched, unmatched_ref, unmatched_pred = fuzzy_match(ref_list, pred_list, threshold=threshold_value)
            #print(f"Prediction: {pred_list} Matched: {matched}")
            # Ensure unmatched_ref and unmatched_pred are lists (if fuzzy_match returns dicts)
            unmatched_ref = list(unmatched_ref.values()) if isinstance(unmatched_ref, dict) else unmatched_ref
            unmatched_pred = list(unmatched_pred.values()) if isinstance(unmatched_pred, dict) else unmatched_pred
            # Combine matched elements with unmatched ones for F1 calculation
            updated_ref_list = list(matched.values()) + unmatched_ref if isinstance(matched, dict) else matched + unmatched_ref
            updated_pred_list = list(matched.values()) + unmatched_pred if isinstance(matched, dict) else matched + unmatched_pred
            precision, recall, f1 = compute_f1_score(updated_ref_list, updated_pred_list)
            frame_f1_scores_od.append((precision, recall, f1))





                # Extract precision, recall, and f1 from the tuples
        _, precisions_lus, recalls_lus, f1s_lus = zip(*frame_f1_scores_lus) if frame_f1_scores_lus else ([], [], [], [])

        if f1s_lus:
            avg_precision = sum(precisions_lus) / len(precisions_lus)
            avg_recall = sum(recalls_lus) / len(recalls_lus)
            avg_f1 = sum(f1s_lus) / len(f1s_lus)
            sentence_f1_scores_lus.append((avg_precision, avg_recall, avg_f1))


        f1_scores_only = [f1 for _, _, f1 in frame_f1_scores]
        if f1_scores_only:
            sen_sum = sum(f1_scores_only) / len(f1_scores_only)
            sentence_f1_scores.append(sen_sum)

               # Extract precision, recall, and f1 from the tuples
        precisions_roles, recalls_roles, f1s_roles = zip(*frame_f1_scores_roles) if frame_f1_scores_roles else ( [], [], [])

        if f1s_roles:
            avg_precision = sum(precisions_roles) / len(precisions_roles)
            avg_recall = sum(recalls_roles) / len(recalls_roles)
            avg_f1 = sum(f1s_roles) / len(f1s_roles)
            sentence_f1_scores_roles.append((avg_precision, avg_recall, avg_f1))

        f1_scores_only_span = [f1 for _, _, f1 in frame_f1_scores_span]
        if f1_scores_only_span:
            sen_sum = sum(f1_scores_only_span) / len(f1_scores_only_span)
            sentence_f1_scores_span.append(sen_sum)

        precisions_ow, recalls_ow, f1s_ow = zip(*frame_f1_scores_ow) if frame_f1_scores_ow else ( [], [], [])

        if f1s_ow:
            avg_precision = sum(precisions_ow) / len(precisions_ow)
            avg_recall = sum(recalls_ow) / len(recalls_ow)
            avg_f1 = sum(f1s_ow) / len(f1s_ow)
            sentence_f1_scores_ow.append((avg_precision, avg_recall, avg_f1))

        precisions_od, recalls_od, f1s_od = zip(*frame_f1_scores_od) if frame_f1_scores_od else ( [], [], [])

        if f1s_od:
            avg_precision = sum(precisions_od) / len(precisions_od)
            avg_recall = sum(recalls_od) / len(recalls_od)
            avg_f1 = sum(f1s_od) / len(f1s_od)
            sentence_f1_scores_od.append((avg_precision, avg_recall, avg_f1))

    # Compute final mean precision, recall, and F1 score across all sentences
    if sentence_f1_scores_lus:
        precisions, recalls, f1s = zip(*sentence_f1_scores_lus)
        final_mean_precision = sum(precisions) / len(precisions)
        final_mean_recall = sum(recalls) / len(recalls)
        final_mean_f1_lu = f1 = 2 * final_mean_precision * final_mean_recall / (final_mean_precision + final_mean_recall)
    else:
        final_mean_precision = final_mean_recall = final_mean_f1 = 0

    final_mean_f1_fes = sum(sentence_f1_scores) / len(sentence_f1_scores) if sentence_f1_scores else 0

    if sentence_f1_scores_roles:
        precisions, recalls, f1s = zip(*sentence_f1_scores_roles)
        final_mean_precision_roles = sum(precisions) / len(precisions)
        final_mean_recall_roles = sum(recalls) / len(recalls)
        final_mean_f1_roles = f1 = 2 * final_mean_precision_roles * final_mean_recall_roles / (final_mean_precision_roles + final_mean_recall_roles)
    else:
        final_mean_precision = final_mean_recall = final_mean_f1 = 0
    final_mean_f1_span = sum(sentence_f1_scores_span) / len(sentence_f1_scores_span) if sentence_f1_scores_span else 0

    if sentence_f1_scores_ow:
        precisions, recalls, f1s = zip(*sentence_f1_scores_ow)
        final_mean_precision_ow = sum(precisions) / len(precisions)
        final_mean_recall_ow = sum(recalls) / len(recalls)
        final_mean_f1_ow = f1 = 2 * final_mean_precision_ow * final_mean_recall_ow / (final_mean_precision_ow + final_mean_recall_ow)
    else:
        final_mean_precision_ow = final_mean_recall_ow = final_mean_f1_od = 0

    if sentence_f1_scores_od:
        precisions, recalls, f1s = zip(*sentence_f1_scores_od)
        final_mean_precision_od = sum(precisions) / len(precisions)
        final_mean_recall_od = sum(recalls) / len(recalls)
        final_mean_f1_od = f1 = 2 * final_mean_precision_od * final_mean_recall_od / (final_mean_precision_od + final_mean_recall_od)
    else:
        final_mean_precision_od = final_mean_recall_od = final_mean_f1_od = 0

    print(f"Calculating F1 Score......")
    print(f"Lexical Units ==> Percision: {final_mean_precision:.4f}, Recall: {final_mean_recall:.4f}, F1: {final_mean_f1_lu:.4f}")
    print(f"Frame Elements: {final_mean_f1_fes:.4f}")
    print(f"Frame Elements Roles Only ==> Percision: {final_mean_precision_roles:.4f}, Recall: {final_mean_recall_roles:.4f}, F1: {final_mean_f1_roles:.4f}")
    print(f"Frame Elements Spans: {final_mean_f1_span:.4f}")
    print(f"Frame Elements One Word Matched ==> Percision: {final_mean_precision_ow:.4f}, Recall: {final_mean_recall_ow:.4f}, F1: {final_mean_f1_ow:.4f}")
    print(f"Frame Elements {threshold_value}% matched ==> Percision: {final_mean_precision_od:.4f}, Recall: {final_mean_recall_od:.4f}, F1: {final_mean_f1_od:.4f}")
    return {"lexical_units_f1": final_mean_f1_lu, "frame_elements_f1": final_mean_f1_fes, "roles_precision": final_mean_precision_roles, "roles_recall": final_mean_recall_roles, "roles_f1": final_mean_f1_roles, "spans_f1": final_mean_f1_span, "one_word_precision": final_mean_precision_ow, "one_word_recall": final_mean_recall_ow, "one_word_f1": final_mean_f1_ow, "overlap25_precision": final_mean_precision_od, "overlap25_recall": final_mean_recall_od, "overlap25_f1": final_mean_f1_od}
