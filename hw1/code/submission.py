import json
import collections
import argparse
import random
import numpy as np


from util import *

random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    
    # Initialize the feature dictionary
    features = {}

    # Combine the words from sentence1 and sentence2
    combined_sentences = ex['sentence1'] + ex['sentence2']
    
    # Iterate over each word, including punctuation
    for word in combined_sentences:
        # word = word.lower()  # Convert to lowercase to handle case-insensitivity
        if word in features:
            features[word] += 1
        else:
            features[word] = 1
  
    return features
    # END_YOUR_CODE

'''
def extract_custom_features(ex):
    """Return unigram TF-IDF features from the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list of str), and sentence2 (list of str)
    Returns:
        A dictionary of unigram TF-IDF features.
    """
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    import string
    features = {}
    translator = str.maketrans('', '', string.punctuation)

    # clean the two sentences
    sentence1 = [word.translate(translator).lower() for word in ex['sentence1']]
    sentence2 = [word.translate(translator).lower() for word in ex['sentence2']]

    tf_sentence1 = {}
    tf_sentence2 = {}

    for word in sentence1:
        tf_sentence1[word] = tf_sentence1.get(word, 0) + 1
    total_words_sentence1 = len(sentence1)
    for word in tf_sentence1:
        tf_sentence1[word] = tf_sentence1[word] / total_words_sentence1

    for word in sentence2:
        tf_sentence2[word] = tf_sentence2.get(word, 0) + 1
    total_words_sentence2 = len(sentence2)
    for word in tf_sentence2:
        tf_sentence2[word] = tf_sentence2[word] / total_words_sentence2

    idf = {}
    num_docs = 2

    words_in_sentence1 = set(sentence1)
    words_in_sentence2 = set(sentence2)

    for word in words_in_sentence1.union(words_in_sentence2):
        doc_count = 0
        if word in words_in_sentence1:
            doc_count += 1
        if word in words_in_sentence2:
            doc_count += 1
        idf[word] = math.log(num_docs / doc_count)

    for word, tf_val in tf_sentence1.items():
        features[word] = tf_val * idf.get(word, 0)

    for word, tf_val in tf_sentence2.items():
        if word in features:
            features[word] += tf_val * idf.get(word, 0)
        else:
            features[word] = tf_val * idf.get(word, 0)

    
    return features

    # END_YOUR_CODE
'''
# n grams

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    features = {}
    import string

    # Clear punctuation
    # translator = str.maketrans('', '', string.punctuation)
    
    # add the two sentences
    combined_sentences = ex['sentence1'] + ex['sentence2']
    # cleaned_words = [word.translate(translator).lower() for word in combined_sentences]

    # extract unigram features
    for word in combined_sentences:
        features[word] = features.get(word, 0) + 1

    # extract bigram features
    bigrams = zip(combined_sentences, combined_sentences[1:])
    for bigram in bigrams:
        bigram_str = ' '.join(bigram)
        features[bigram_str] = features.get(bigram_str, 0) + 1

    return features
    # END_YOUR_CODE



def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    weights = {}

    for eppoch in range(num_epochs):
        for ex in train_data:
            features = feature_extractor(ex)

            # True label
            y = ex['gold_label'] 

            # Prediction
            y_hat = predict(weights, features)

            error = y - y_hat
            increment(weights, features, learning_rate * error)

    return weights 
    # END_YOUR_CODE

def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    # BEGIN_YOUR_CODE
    vocab = list(set(tokens))
    vocab_size = len(vocab)

    word2ind = {word: i for i, word in enumerate(vocab)}

    co_mat = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for i, word in enumerate(tokens):
        word_idx = word2ind[word]
        
        # Find the neighbour of each word
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # loop
        for j in range(start, end):
            if i != j:  # take out the case where the variable corresponds to itself
                context_word = tokens[j]
                context_word_idx = word2ind[context_word]
                co_mat[word_idx, context_word_idx] += 1
                
    return word2ind, co_mat

    # END_YOUR_CODE

def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    # Perform SVD
    U, S, Vt = np.linalg.svd(co_mat, full_matrices=False, hermitian = True)

    # Adjust embed_size to not exceed the number of singular values
    embed_size = min(embed_size, U.shape[1], S.shape[0])
    
    # Take the first 'embed_size' dimensions from U and scale by the singular values
    embeddings = U[:, :embed_size] * S[:embed_size]
    

    return embeddings

    # END_YOUR_CODE

def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='cosine'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    # BEGIN_YOUR_CODE
    target_embedding = embeddings[word_ind]

    # calculate the metric
    if metric == 'dot':
        # distance
        similarities = np.dot(embeddings, target_embedding)
    elif metric == 'cosine':
        # cosine similarity
        target_norm = np.linalg.norm(target_embedding)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        similarities = np.dot(embeddings, target_embedding) / (embeddings_norm * target_norm)
    else:
        raise ValueError("Unknown metric: choose 'dot' or 'cosine'")
    
    similarities[word_ind] = -np.inf
    
    # find top k popular similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    ind2word = {ind: word for word, ind in word2ind.items()}
    topk_words = [ind2word[i] for i in top_k_indices]
    
    return topk_words

    # END_YOUR_CODE
