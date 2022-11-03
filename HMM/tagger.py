import numpy as np
from hmm import HMM
from collections import defaultdict
def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    i = 0 
    for idx, val in enumerate(unique_words):
        if val not in word2idx:
            word2idx[val] = i
            i += 1
            
    for i in range(0, S):
        tag2idx[tags[i]] = i


    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    first_tag_dict, count_tags, total_transitions = defaultdict(int), defaultdict(int), defaultdict(int)
    word_pos, n_trans_tag = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))

    for line in train_data:
        prev_tag = ""
        temp = line.tags[0]
        first_tag_dict[temp] = first_tag_dict[temp]+1

        for i, word in enumerate(line.words):
            count_tags[line.tags[i]] += 1
            word_pos[line.tags[i]][word] += 1
            total_transitions[prev_tag] += 1
            n_trans_tag[prev_tag][line.tags[i]] += 1
            prev_tag = line.tags[i]

    
    for tag in tags:
        tag_index = tag2idx[tag]
        pi[tag_index] = first_tag_dict[tag] / len(train_data)
        for word, count in word_pos[tag].items():
            B[tag_index][word2idx[word]] = count / count_tags[tag]
        for next_tag in tags:
            A[tag_index][tag2idx[next_tag]] = n_trans_tag[tag][next_tag] / total_transitions[tag]


    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    er, idx = np.full((len(tags), 1), 10 ** -6), max(model.obs_dict.values())+1
    for line in test_data:
        for w in line.words:
            if w not in model.obs_dict:
                model.B = np.append(model.B, er, axis=1)
                model.obs_dict[w] = idx
                idx += 1
        p = model.viterbi(line.words)
        tagging.append(p)
        
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
