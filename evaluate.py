import math
from collections import deque
from copy import copy
import os
from hmm_model_builder import *
import nltk
import numpy as np
import pickle


# Extracts each word's stem stored in a specific file and stores the word stem mapping in a dictionary 
def lemmatize_words(file_path:str, word_to_stem:dict):
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.replace("\n", "")

            line = line.split(sep='\t')

            data_rows.append(line)
    for row in data_rows:
        if row[1] == '' or row[1] == ' ' or (len(row[1]) ==1 and ord(row[1]) == 0):
            row[1] = row[0]
        word_to_stem[row[0].lower()] = row[1].lower() 
        
# returns a dictionary mapping each word in the corpus to its stem 
def create_word_stem_mapping(directory_path:str, word_to_stem:dict)->str:
    try:
       for filename in os.listdir(directory_path):
           file_path = os.path.join(directory_path, filename)
           try:
               lemmatize_words(file_path, word_to_stem)
           except Exception as e:
               print(e)
               print(f"Issue lemmatizing text file")

    except Exception as e:
        print(f"Bad access on directory: {str(e)}")

    return word_to_stem

# Saves dictionary with word to stem mapping
def save_word_stem_mapping(word_to_stem, output_file_path):
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(word_to_stem, f)
        print(f"Successfully saved mapping to {output_file_path}")
    except Exception as e:
        print(f"Error saving mapping: {str(e)}")

# Loads the dictionary
def load_word_stem_mapping(input_file_path):
    try:
        with open(input_file_path, 'rb') as f:
            word_to_stem = pickle.load(f)
        return word_to_stem
    except Exception as e:
        print(f"Error loading mapping: {str(e)}")
        return None 
        
input_dict = defaultdict(str)

# get 
def get_word_pos(lemma_file: str, pos_tag_file: str) -> dict:

    #dictionary storing each lemmatized word and its corresponding part-of-speech(pos)
    word_pos = {}
    #dictionary storing each lemmatized word and its corresponding frequency in the lemma file
    word_freq = defaultdict(int)
    with open(lemma_file, 'r') as f1:
        with open(pos_tag_file, 'r') as f2:
            # reads both lemma and pos files
            lemma_text = f1.read()
            pos_text = f2.read()

            lemma_lines = lemma_text.split('+')
            pos_lines = pos_text.split('+')
            lemma_sentence = ""
            pos_sentence = ""

            # Iterates over each sentence in lemma and pos files
            for i in range(len(lemma_lines)):
                lemma_sentence = lemma_lines[i]
                pos_sentence = pos_lines[i]

                lemma_words = lemma_sentence.split()
                pos_words = pos_sentence.split()

                # Iterates over every word in the current lemma sentence and maps each word to its corresponding pos as well as each word with its frequency in the lemma file
                for i in range(len(lemma_words)):
                    if lemma_words[i] not in word_pos:
                        word_pos[lemma_words[i]] = pos_words[i]
                    word_freq[lemma_words[i]] = word_freq.get(lemma_words[i], 0)+1
    return word_pos

def perplexity(model : dict, corpus : str):
    prob =  model_likelihood(model, corpus)
    if prob == 0:
        return "Undefined"
    perplexity = prob**(-1/len(corpus.split()))
    return perplexity


def unscramble(scrambled_sentence, use_mle):

    # builds hmm model with mle/greedy
    model = build_model(use_mle, False)
    scrambled_sentence_with_eos = '+ '+scrambled_sentence+' +'
    print(f"Scrambled sentence: {scrambled_sentence}")
    scrambled_perplexity = perplexity(model, scrambled_sentence_with_eos)
    print(f"Perplexity of scrambled sentence: <{scrambled_perplexity}")

    word_pos = get_word_pos('train_corpus.txt', 'train_corpus_pos.txt')
    word_to_stem = load_word_stem_mapping('word_stem_mapping.txt')
    word_to_stem['+'] = '+'
    wnl = nltk.WordNetLemmatizer()
    remaining_words = defaultdict(str)
    remaining_words_freq = defaultdict(int)
    scrambled_words = scrambled_sentence.lower()
    word_pos['i'] = 'PRP$'

    # stem each word in scrambled sentence
    stem_to_original = defaultdict(str)
    scrambled_words = scrambled_sentence.split()
    for i in range(len(scrambled_words)):
        original_word = copy(scrambled_words[i])
        if scrambled_words[i] in word_to_stem:
            scrambled_words[i] = word_to_stem[scrambled_words[i]]
        else:
            scrambled_words[i] = wnl.lemmatize(scrambled_words[i])


        stem_to_original[scrambled_words[i]] = original_word
    stem_to_original['+'] = '+'

    # length of beam used in search
    beam_len = math.floor(math.sqrt(len(scrambled_words)))+1

    for w in scrambled_words:
        if w not in word_pos:
            (_,tag) = nltk.pos_tag([wnl.lemmatize(w)])[0]
            remaining_words[w] = tag
        else:
            remaining_words[w] = word_pos[w]
        remaining_words_freq[w] = remaining_words_freq.get(w,0)+1

    ''' 
     The state being represented is composed of the following: 
     current_pos - The pos of the last observation token in the current sequence. For eg: in the sequence '+ the', current_pos would be 'DET' 
     curr_seq - The current sequence formed 
     remaining_words - a dict mapping each word to its pos
     remaining_words_freq - a dict mapping each word to its frequency in scrambled string
    '''

    # Initializes starting state
    initial_state = ('+', '+', 1, remaining_words, remaining_words_freq)



    (curr_pos, curr_seq, curr_joint_prob, remaining_words, remaining_words_freq) = initial_state

    next_states_and_prob = defaultdict(float)


    # stores the states to be expanded in beam search
    queue = deque()
    queue.append(initial_state)
    # highest prob sequence with length equal to length of scrambled sentence
    best_seq = ""

    # Beam Search
    while queue:
        all_states = []
        max_prob = 0
        for _ in range(len(queue)):
            # polls state from queue
            curr_pos, curr_seq, curr_joint_prob, remaining_words_with_pos, remaining_words_freq = queue.popleft()
            seq_length = len(curr_seq.split())
            # checks if the number of words excluding '+' equals the length of the scrambled string to know if a full sentence has been formed
            # updates max probability and best seq
            if seq_length-1 == len(scrambled_words):
                if curr_joint_prob > max_prob:
                    max_prob = curr_joint_prob
                    best_seq = copy(curr_seq)

            # gets node value for this pos
            curr_node = model[curr_pos]

            # Iterates over all potential next states(pos's) from this pos/state and stores the state/pos along with the transition probability from this state to the next one in a dictionary
            for next_state in curr_node.transitions.keys():
                next_states_and_prob[next_state] = curr_node.get_transition_probability(next_state)

            # Iterates over each next state/pos and checks if any of the remaining words in the state have pos equal to next_state
            # If so, the word is used to form a new sequence and the remaining_words and remaining_words_freq is updated
            for next_state in next_states_and_prob.keys():
                for k, _ in remaining_words_freq.items():
                    pos = remaining_words_with_pos[k]
                    if pos == next_state:
                        new_seq =  curr_seq+' '+k

                        # joint probability with extra weightage assigned to pos transitions(want to increase importance of pos transitions)
                        new_join_prob = curr_joint_prob*next_states_and_prob[pos]**1.4*model[pos].get_emission_probability(k)**0.8
                        new_remaining_words = remaining_words.copy()
                        new_remaining_words_freq = remaining_words_freq.copy()
                        new_remaining_words_freq[k] = new_remaining_words_freq[k]-1
                        # Removes word if the count of k goes to zero
                        if new_remaining_words_freq[k] == 0:
                            del new_remaining_words[k]
                            del new_remaining_words_freq[k]
                        # Add all expanded states to all_states
                        all_states.append((pos, new_seq, new_join_prob, new_remaining_words, new_remaining_words_freq))
                        break

        # sorts all states in descending order using joint prob of new sequence(o) formed and its pos's(s), ie P(s,o)
        all_states.sort(key= lambda x:x[2], reverse= True)
        beam_len = min(len(all_states), beam_len)
        queue.clear()
        queue.extend(all_states[:beam_len])

    unscrambled_sentence = ' '.join([stem_to_original[word] for word in best_seq.split() if word != '+'])
    unscrambled_sentence_with_eos = '+ '+unscrambled_sentence+' +'
    unscrambled_perplexity = perplexity(model, unscrambled_sentence_with_eos)

    print(f"Unscrambled sentence: {unscrambled_sentence}")

    if use_mle:
        print(f"Perplexity of unscrambled sentence using mle model: <{unscrambled_perplexity}>")
    else:
        print(f"Perplexity of unscrambled sentence using greedy model: <{unscrambled_perplexity}>")


    return unscrambled_sentence

# sets up command line parser
def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Language Model Operations')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    perplexity_parser = subparsers.add_parser('evaluate', help='Calculate perplexity')
    perplexity_parser.add_argument('--model-type', type=str, choices=['ngram', 'hmm'],
                                   required=True, help='Type of model to use')

    perplexity_parser.add_argument('--n', type=int, help='N-gram size (required for ngram model)')
    perplexity_parser.add_argument('--smoothing', action='store_true',
                                   help='Use smoothing (for ngram model)')
    perplexity_parser.add_argument('--mle', action='store_true',
                                   help='Use MLE estimation (for HMM model)')
    perplexity_parser.add_argument('test_file', help='Test data file path')

    unscramble_parser = subparsers.add_parser('unscramble', help='Unscramble text')
    unscramble_parser.add_argument('--model-type', type=str, choices=['ngram', 'hmm'],
                                   required=True, help='Type of model to use')
    unscramble_parser.add_argument('--n', type=int, help='N-gram size (required for ngram model)')
    unscramble_parser.add_argument('--smoothing', action='store_true',
                                   help='Use smoothing (for ngram model)')
    unscramble_parser.add_argument('--mle', action='store_true',
                                   help='Use MLE estimation (for HMM model)')
    unscramble_parser.add_argument('scrambled_file', help='File containing scrambled text')

    return parser


# evaluate perplexity wrapper for ngram and hmm models
def evaluate_perplexity(args):
    """Wrapper function to handle perplexity calculation for both models"""
    if args.model_type == 'ngram':
        if args.n is None:
            raise ValueError("N-gram size (-n) is required for ngram model")
        calculate_perplexity(
            n=args.n,
            use_smoothing=args.smoothing,
            test_data_file=args.test_file
        )
    else:
        with open(args.test_file, 'r') as file:
            test_text = file.read()
        model = build_model(args.mle)
        result = perplexity(model, test_text)
        print(f"HMM Model Perplexity: <{result}>")

#unscramble wrapper for ngram and hmm models
def handle_unscramble(args):
    if args.model_type == 'ngram':
        if args.n is None:
            raise ValueError("N-gram size (-n) is required for ngram model")
        unscramble(
            n=args.n,
            use_smoothing=args.smoothing,
            scrambled_file=args.scrambled_file
        )
    else:
        with open(args.scrambled_file, 'r') as file:
            scrambled_text = file.read()
        unscramble(scrambled_text, args.mle)


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate_perplexity(args)
    elif args.command == 'unscramble':
        handle_unscramble(args)



