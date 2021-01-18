"""
PIDS - BERT
-----------
Testing the extent to which physical inferences made by humans
in the course of resolving ambiguous pronouns can be predicted
by language models (here, BERT).

"""

import logging
import re

from torch import tensor, argsort, softmax, no_grad
import numpy as np
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForMaskedLM


"""
Setup
-----
Logging & parameters

"""

logging.basicConfig(level=logging.INFO)

USE_GPU = 1

"""
Model setup
-----------
Load BERT tokenizer & BERT for Masked LM and evaluate.
"""

# Load pre-trained model tokenizer (vocabulary)
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# Load pre-trained model (weights)
MODEL = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
MODEL.eval()


"""
Inference functions
-------------------
Pass texts to model and get probabilities of different completions.
"""


def predict_masked(text, n=5, model=MODEL, tokenizer=TOKENIZER):
    """Top n predictions for all masked words in text

    Args:
        text (str): Text containing masks
        n (int, optional): Number of candidates to produce (5)
        model (TYPE, optional): language model (BERT)
        tokenizer (TYPE, optional): lm tokenizer (BERT)

    Returns:
        candidates: List of tuples  of (index, [(candidate, prob)])
    """
    tokenized_text = tokenizer.tokenize(text)
    mask_inds = np.where(np.array(tokenized_text) == "[MASK]")

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = tensor([indexed_tokens])

    # Predict all tokens
    with no_grad():
        outputs = model(tokens_tensor)  # token_type_ids=segments_tensors)
        predictions = outputs[0]

    # get predicted tokens
    out = []
    for mask in mask_inds[0]:
        print("Predicting mask index: ", mask)

        # prediction for mask
        mask_preds = predictions[0, mask.item()]
        mask_preds = softmax(mask_preds, 0)
        predicted_indices = [x.item() for x in
                             argsort(mask_preds, descending=True)[:n]]
        scores = [mask_preds[i].item() for i in predicted_indices]
        predicted_tokens = []
        for index in predicted_indices:
            predicted_tokens.append(tokenizer.convert_ids_to_tokens([index])[0])
        out.append((mask, list(zip(predicted_tokens, scores))))

    return out


text = "[CLS] Sally hit Jenny because [MASK] was angry. [SEP]"

text = "[CLS] When the steel plate fell on the glass plate, [MASK] [MASK] [MASK] broke. [SEP]"


text = "[CLS] When chloe threw the rock at the egg, the [MASK] cracked. [SEP]"

predict_masked(text, 20)

text = "[CLS] When chloe threw the egg at the rock, the [MASK] cracked. [SEP]"

predict_masked(text, 20)


def mask_probability(text, candidates, model=MODEL, tokenizer=TOKENIZER):
    """Probabilities for candidates as replacement for [MASK] in text

    Args:
        text (str): Text containing a single [MASK]
        candidates (list of str): Candidate mask replacements
        model (TYPE, optional): language model (BERT)
        tokenizer (TYPE, optional): lm tokenizer (BERT)

    Returns:
        candidates (dict): {candidate: prob}
    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    # Get candidate ids
    candidate_ids = {}
    for candidate in candidates:
        candidate_tokens = candidate.split()
        candidate_ids[candidate] = tokenizer.convert_tokens_to_ids(
            candidate_tokens)

        # TODO: Check for 100 tokens

    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate, ids, in candidate_ids.items():

        # Add a mask for each token in candidate

        candidate_text = re.sub("\[MASK\] ", "[MASK] " * len(ids), text)

        # Tokenize text
        tokenized_text = tokenizer.tokenize(candidate_text)
        mask_inds = np.where(np.array(tokenized_text) == "[MASK]")

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = tensor([indexed_tokens])

        # Predict all tokens
        with no_grad():
            outputs = model(tokens_tensor)  # token_type_ids=segments_tensors)
            predictions = outputs[0]

        # get predicted tokens
        probs = []

        for (i, mask) in enumerate(mask_inds[0]):
            # prediction for mask
            mask_preds = predictions[0, mask.item()]
            mask_preds = softmax(mask_preds, 0)
            prob = mask_preds[ids[i]].item()
            probs.append(np.log(prob))

        candidate_probs[candidate] = np.exp(np.mean(probs))

    return candidate_probs


text = "[CLS] When chloe threw the egg at the rock, the [MASK] cracked. [SEP]"
candidates = ["egg", "rock"]

mask_probability(text, candidates)

text = "[CLS] When chloe threw the rock at the egg, the [MASK] cracked. [SEP]"
candidates = ["egg", "rock"]

mask_probability(text, candidates)

# Glass vs steel

text = "[CLS] When the steel plate fell on the glass plate, the [MASK] plate broke. [SEP]"
candidates = ["glass", "steel"]

mask_probability(text, candidates)

text = "[CLS] When the glass plate fell on the steel plate, the [MASK] plate broke. [SEP]"
candidates = ["glass", "steel"]

mask_probability(text, candidates)

# Multi-word

text = "[CLS] When the glass plate fell on the steel plate, [MASK] broke. [SEP]"
candidates = ["the glass plate", "the steel plate"]

mask_probability(text, candidates)



"""
Apply to csv
------------
"""

stimuli = pd.read_csv("stims.csv")

np1_probs = []
np2_probs = []

for i, row in stimuli.iterrows():

    # Extract text
    text = row['text']

    # Enforce CLS and SEP tags
    if text[:5] != "[CLS]":
        text = "[CLS] " + text

    if text[-5:] != "[SEP]":
        text = text + " [SEP]"

    # Extract candidates
    np1 = row['np1']
    np2 = row['np2']
    candidates = [np1, np2]

    # Get probs
    probs = mask_probability(text, [np1, np2])

    # Add to lists
    np1_probs.append(probs[np1])
    np2_probs.append(probs[np2])

    print("\n\n" + "-" * 30 + "\n\n")
    print(f"{i} / {len(stimuli)}: {text}")
    for k, v in probs.items():
        print(f"{k}: {v:.3g}")

stimuli["np1_prob"] = np1_probs
stimuli["np2_prob"] = np2_probs

stimuli.to_csv("stimuli.csv")