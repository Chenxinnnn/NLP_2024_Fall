import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    neighbors = {
    'a': ['q', 's', 'z'], 'e': ['w', 'r', 'd'], 'i': ['u', 'o', 'k'], 
    'o': ['i', 'p', 'l'], 'u': ['y', 'i', 'j'], 'y': ['t', 'u', 'h'],
    't': ['r', 'y', 'g'], 'r': ['e', 't', 'f'], 's': ['a', 'd', 'w'],
    'd': ['s', 'f', 'e'], 'f': ['d', 'g', 'r'], 'g': ['f', 'h', 't'],
    'h': ['g', 'j', 'y'], 'j': ['h', 'k', 'u'], 'k': ['j', 'l', 'i'],
    'l': ['k', 'o', 'p'], 'p': ['o', 'l', ';'],
}
    # Tokenize the text into words
    words = example["text"].split()
    
    #define a function to introduce typos in the word
    def typos(word, near_keys):
        if random.random() < 0.4:
            selected_letter = random.choice(word)
            if selected_letter in near_keys:
                replaced_letter = random.choice(near_keys[selected_letter])
                word = word.replace(selected_letter, replaced_letter, 1)
        return word

    transformed_words = [typos(word, neighbors) for word in words]

    # Join transformed words back into a sentence
    example["text"] = " ".join(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
