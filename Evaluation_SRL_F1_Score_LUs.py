#Lemmatized Lexical Units
#Calculate F1 Score for Lexical units, Frame Elements, Roles, Spans, One Word Matching
# Remove sentences from Gold Standars if they are not avialable in Predictions
# Calculate F1 score at Sentence and Frame level individulay
import json
import nltk
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

# POS tag dictionary for WordNet
POS_TAG_MAP = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

# Cache for lexical unit normalization
LEXICAL_UNIT_CACHE = {}

def get_wordnet_pos(word):
    """
    Map NLTK POS tag to the corresponding WordNet POS for lemmatization.
    """
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    return POS_TAG_MAP.get(pos_tag, wordnet.NOUN)  # Default to NOUN if not found

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

# POS tag map for converting to WordNet format
POS_TAG_MAP = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

# Cache to avoid reprocessing
LEXICAL_UNIT_CACHE = {}

def get_wordnet_pos(pos_tag):
    """
    Convert a POS tag (from nltk.pos_tag) to WordNet POS format.
    """
    return POS_TAG_MAP.get(pos_tag[0].upper(), wordnet.NOUN)  # Default to NOUN

def normalize_lexical_unit_in_sentence(lu, sentence):
    """
    Normalize the lexical unit using full sentence POS tagging.
    """
    lu = lu.lower().strip()

    if lu in LEXICAL_UNIT_CACHE:
        return LEXICAL_UNIT_CACHE[lu]

    # Tokenize and tag the whole sentence
    tokens = nltk.word_tokenize(sentence)
    pos_tags = dict(nltk.pos_tag(tokens))

    words = lu.split()
    normalized_words = []

    for word in words:
        word_pos = pos_tags.get(word, "NN")  # Fallback to noun if not found
        wordnet_pos = get_wordnet_pos(word_pos)
        normalized_words.append(lemmatizer.lemmatize(word, wordnet_pos))

    normalized_lu = " ".join(normalized_words)
    LEXICAL_UNIT_CACHE[lu] = normalized_lu
    return normalized_lu


def extract_frames_and_lus(annotations, sentence):
    frames_lus = {}
    frames = []

    for ann in annotations:
        fram = ann.get("frame", "").strip().lower()
        frame = fram.replace("_", " ")
        lu = ann.get("lexical_unit", "").strip()
        frames.append(frame)

        if frame and lu:
            normalized_lu = normalize_lexical_unit_in_sentence(lu, sentence)
            if frame not in frames_lus:
                frames_lus[frame] = []
            frames_lus[frame].append(normalized_lu)

    return frames_lus, frames


STOPWORDS = set(["a", "an", "the", "in", "on", "at", "and", "of", "to", "is", "that", "it", "for", "with", "as", "was", "were", "by", "this", "which", "or", "be", "are"])
def extract_frames_and_lus_1(annotations):

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

STOPWORDS = set(stopwords.words("english"))

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


def one_word_found(ref_list, pred_list):
    """Match reference and predicted items if at least one word from reference exists anywhere in prediction."""
    matched = []
    unmatched_ref = list(ref_list)
    unmatched_pred = list(pred_list)
    match_details = []

    for ref in ref_list:
        ref_words = set(ref[1].split()) - STOPWORDS
        ref_role, ref_argument = ref

        for pred in pred_list:
            pred_words = set(pred[1].split()) - STOPWORDS
            pred_role, pred_argument = pred

            if any(word in pred_words for word in ref_words):
                matched.append((ref_role, pred_argument))
                if ref in unmatched_ref:
                    unmatched_ref.remove(ref)
                if pred in unmatched_pred:
                    unmatched_pred.remove(pred)
                matched_word = next(word for word in ref_words if word in pred_words)  # Get first matched word
                match_details.append((ref_role, ref_argument, pred_argument, matched_word))
                break

    return matched, unmatched_ref, unmatched_pred, match_details

def compute_f1_score(true_list, pred_list):
    """Compute Precision, Recall, and F1 Score with support for duplicate frame elements."""

    # Convert lists to multisets (counts of each element)
    true_counts = Counter(true_list)
    pred_counts = Counter(pred_list)

    # Collect all unique frame elements
    all_elements = set(true_counts.keys()).union(set(pred_counts.keys()))

    # Construct y_true and y_pred based on counts
    y_true = []
    y_pred = []

    for element in all_elements:
        # Compare the number of occurrences
        true_freq = true_counts[element]
        pred_freq = pred_counts[element]

        # Append 1 as many times as the count of the element
        y_true.extend([1] * true_freq + [0] * (pred_freq - true_freq))
        y_pred.extend([1] * pred_freq + [0] * (true_freq - pred_freq))

    # Compute Precision, Recall, and F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return precision, recall, f1


# Load JSON data
with open('Results/SRL/adapted_srl_result_LUs.json', 'r', encoding='utf-8') as f:
    reference = json.load(f)

with open('Results/SRL/adapted_srl_gold-std_LUs.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

print(f"Reference Sentences: {len(reference)}")
print(f"Predictions Sentences: {len(predictions)}")


# List to store sentence-level F1 scores

sentence_f1_scores_lus = []
sentence_f1_scores_lu = []


# Process each input instance
for ref, pred in zip(reference, predictions):
    ref_sentences = ref.get("input_sentence", [])  # Extract sentences
    pred_sentences = pred.get("input_sentence", [])  # Ensure predictions are aligned
    #print(f"{ref_sentences[:20]}: {pred_sentences[:20]}")

    ref_annotations = ref.get("annotations", [])
    pred_annotations = pred.get("annotations", [])
    ref_lu, ref_frame = extract_frames_and_lus_1(ref_annotations)
    pred_lu, pred_frame = extract_frames_and_lus_1(pred_annotations)

    
    ref_lus, ref_frames = extract_frames_and_lus(ref["annotations"], ref_sentences)
    pred_lus, pred_frames = extract_frames_and_lus(pred["annotations"], pred_sentences)

    #print(f"{ref_frames}: {pred_frames}")

    frame_f1_scores_lus = []
    frame_f1_scores_lu = []


    # Compute F1 for each frame in lexical units
    for frame in set(ref_lus.keys()).union(set(pred_lus.keys())):
        ref_list = ref_lus.get(frame, [])
        pred_list = pred_lus.get(frame, [])
        f1_score = compute_f1_score(ref_list, pred_list)
        frame_f1_scores_lus.append(f1_score)
        #print(f"Frame: {f1_score}: {frame}: {ref_list}: {pred_list}")

    # Compute F1 for each frame in lexical units with normalization
    for frame in set(ref_lu.keys()).union(set(pred_lu.keys())):
        ref_list = ref_lu.get(frame, [])
        pred_list = pred_lu.get(frame, [])
        f1_score = compute_f1_score(ref_list, pred_list)
        frame_f1_scores_lu.append(f1_score)
        #print(f"Frame: {f1_score}: {frame}: {ref_list}: {pred_list}")

    


    # Extract only the F1 scores from the tuples
    f1_scores_only_lus = [f1 for _, _, f1 in frame_f1_scores_lus]
    if f1_scores_only_lus:
        sen_sum_lu = sum(f1_scores_only_lus) / len(f1_scores_only_lus)
        sentence_f1_scores_lus.append(sen_sum_lu)
        #print(f"Sentence: {sen_sum_lu}")

    # Extract only the F1 scores from the tuples
    f1_scores_only_lu = [f1 for _, _, f1 in frame_f1_scores_lu]
    if f1_scores_only_lu:
        sen_sum_lu = sum(f1_scores_only_lu) / len(f1_scores_only_lu)
        sentence_f1_scores_lu.append(sen_sum_lu)
        #print(f"Sentence: {sen_sum_lu}")  

  
# Compute final mean F1 score across all sentences
final_mean_f1_lus = sum(sentence_f1_scores_lus) / len(sentence_f1_scores_lus) if sentence_f1_scores_lus else 0
final_mean_f1_lu = sum(sentence_f1_scores_lu) / len(sentence_f1_scores_lu) if sentence_f1_scores_lu else 0


print(f"Lemmatized Lexical Units F1 Score: {final_mean_f1_lus:.4f}")
print(f"Lexical Units F1 Score: {final_mean_f1_lu:.4f}")

