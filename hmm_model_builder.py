import numpy as np
from collections import defaultdict
import pickle

class Transition:
    def __init__(self, source = "", sink = "", probability = 0.0):
        '''
        An edge in the graph between two nodes in the graph --> A transition between two Parts of Speech

        Parameters
        ------------

        source : str, default=""
            The current state / current node the graph

        sink : str, default=""
            The next state / next node in the graph

        probability : float, default=0.0
            The probability the model will transition from source to sink
        '''
        self.source = source
        self.sink = sink
        self.probability = probability

        # For internal use in Baum-Welch algorithm
        self.count = 0


class Emission:
    def __init__(self, token = "", probability = 0.0) -> None:
        '''
        An emission from a certain POS / state in the HMM

        Parameters
        ------------

        token : String, default=""
            The word that can be emitted from this part of speech / HMM state

        probability : float, default=0.0
            The probability the model will emit this token
        '''
        self.token = token
        self.probability = probability

        # For internal use in Baum-Welch algorithm
        self.count = 0


class Node: 
    def __init__(self, pos = "", transitions = {}, emissions = {}):
        '''
        A node / state in a HMM

        Parameters
        --------------

        pos : str, default=""
            The part of speech
        
        transitions : dict, default={}
            A dictionary of Transitions. The possible transitions from this POS to others in the HMM graph

        emissions : dict, default={}
            A dictionary of Emissions. The words from the training corpus that correspond to this pos
        '''
        self.pos = pos
        self.transitions = transitions
        self.emissions = emissions

    def get_transition_probability(self, pos_sink = "") -> float:
        '''Returns transition probability from this POS to another POS'''
        if pos_sink in self.transitions:
            return self.transitions[pos_sink].probability
        return 0
            
    def get_emission_probability(self, token = "") -> float:
        '''Returns emission probability of a token from this POS'''
        if token in self.emissions:
            return self.emissions[token].probability
        return 0


def get_lexicon_counts(pos_corpus : str) -> None:
    '''Saves the sorted counts for each unigram in a txt file.'''
    unigram_counts = {}
    pos = pos_corpus.split()
    for p in pos:
        if p in unigram_counts:
            unigram_counts[p] += 1
        else:
            unigram_counts[p] = 1
    
    # Sort the dictionary by value
    unigram_counts = {k: v for k, v in sorted(unigram_counts.items(), key=lambda item: item[1], reverse=True)}
    with open("hmm_lexicon.txt", 'w') as file:
        for word, count in unigram_counts.items():
            file.write(f"{word} : {count}\n")
        file.close()



def save_model(model : dict, name : str):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path) -> dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    






def forward_algorithm(model : dict, observation_sequence : str) -> np.ndarray:
    observation_sequence = observation_sequence.split()
    num_states = len(model.keys())
    cache = np.zeros(shape=(len(observation_sequence), num_states))

    # Map the model's POS to their index in the cache
    indicies = {}
    for i, pos in enumerate(model.keys()):
        indicies[pos] = i

    # initialize table[0][k] = 0 for all k != '+'
        # and table[0]['+'] = 1
    for i, pos in enumerate(model.keys()):
        if pos == '+':
            cache[0][i] = 1
        else:
            # This is repetitive as the entire cache has only 0's by default, but its important for illustration and readability
            cache[0][i] = 0
            
    # Recursion step
    for i in range (1, len(observation_sequence)): # skip the '+' token
        for j, node in enumerate(model.values()):
            token = observation_sequence[i]
            alpha = 0
            for predecessor_pos, predecessor in model.items():
                if node.pos in predecessor.transitions:
                    alpha += cache[i - 1][indicies[predecessor_pos]] * model[predecessor_pos].get_transition_probability(node.pos)
            alpha = alpha * node.get_emission_probability(token)
            cache[i][j] = alpha
    return cache


def backward_algorithm(model : dict, observation_sequence : str) -> np.ndarray:
    '''
    I wrote this on accident but I'll leave it 
    '''
    observation_sequence = observation_sequence.split()
    num_states = len(model.keys())
    cache = np.zeros(shape=(len(observation_sequence), num_states))

    # Map the model's POS to their index in the cache
    indicies = {}
    for i, pos in enumerate(model.keys()):
        indicies[pos] = i

    # Initialize table[observation_sequence.length - 1][k] = 0 for all k != '+'
        # and table[observation_sequence.length - 1]['+'] = 1
    for i, pos in enumerate(model.keys()):
        if pos == '+':
            cache[len(observation_sequence) - 1][i] = 1
        else:
            # This is repetitive as the entire cache has only 0's by default, but its important for illustration and readability
            cache[len(observation_sequence) - 1][i] = 0


    # Recursion step
    for i in range (len(observation_sequence) - 2, -1, -1):
        for j, node in enumerate(model.values()):
            next_token = observation_sequence[i + 1]
            beta = 0
            for next_node in node.transitions:
                beta += cache[i + 1][indicies[next_node]] * node.get_transition_probability(next_node) * model[next_node].get_emission_probability(next_token)
            cache[i][j] = beta
    return cache


def baum_welch(model : dict, lemmatized_corpus : str, alpha : np.ndarray, beta : np.ndarray):
    '''
    Runs one iteration of the Baum Welch Algorithm
    I wrote this on accident but I'll leave it. 
    '''
    # Reset the model's counts for emissions and transitions
    for pos, node in model.items():
        for emission in node.emissions.values():
            emission.count = 0
        for transition in node.transitions.values():
            transition.count = 0

    
    # Map the model's POS to their index in the cache
    indicies = {}
    for i, pos in enumerate(model.keys()):
        indicies[pos] = i


    lemmatized_corpus = lemmatized_corpus.split()
    # Update the emission counts
    for i, token in enumerate(lemmatized_corpus):
        for pos, node in model.items():
            if token in node.emissions:
                emission = node.emissions[token]
                emission.count += alpha[i][indicies[pos]] * beta[i][indicies[pos]] / alpha[len(lemmatized_corpus) - 1][indicies['+']]
    # Update the emission probabilities
    for node in model.values():
        sum = 0
        for emission in node.emissions.values():
            sum += emission.count
        for emission in node.emissions.values():
            emission.probability = emission.count / sum

    # Update the transition counts
    for i in range (len(lemmatized_corpus) - 1):
        for pos, node in model.items():
            for neighbor_pos, neighbor in model.items():
                if neighbor_pos in node.transitions:
                    transition = node.transitions[neighbor_pos]
                    transition.count += (alpha[i][indicies[pos]] * node.get_transition_probability(neighbor_pos) * neighbor.get_emission_probability(lemmatized_corpus[i + 1]) * beta[i + 1][indicies[neighbor_pos]]) / (alpha[len(lemmatized_corpus) - 1][indicies['+']])

    # Update the transition probabilities
    for node in model.values():
        sum = 0
        for transition in node.transitions.values():
            sum += transition.count
        for transition in node.transitions.values():
            transition.probability = transition.count / sum
    return model





def train_with_mle(model : dict, lemmatized_corpus : list, pos_corpus : list) -> dict:

    for i in range (len(lemmatized_corpus) - 1): 
        current_node = model[pos_corpus[i]]
        transition = current_node.transitions[pos_corpus[i + 1]]
        emission = current_node.emissions[lemmatized_corpus[i]]

        transition.probability += 1
        emission.probability += 1

    # Use (count + 1) smoothing to prevent 0 transition probabilities
    for node in model.values():
        for transition in node.transitions.values():
            transition.probability += 1

    
    # Normalize all the transition and emission probabilities so that they sum to 1
    for node in model.values():
        sum_transition = len(node.transitions.values()) # To account for the smoothing count adjustment
        for transition in node.transitions.values():
            sum_transition += transition.probability
        for transition in node.transitions.values():
            transition.probability = transition.probability / sum_transition

        sum_emission = 0
        for emission in node.emissions.values():
            sum_emission += emission.probability
        for emission in node.emissions.values():
            emission.probability = emission.probability / sum_emission

    return model




def model_likelihood(model : dict, validation_corpus : str) -> float:
    # Map the model's POS to their index in the cache
    indicies = {}
    for i, pos in enumerate(model.keys()):
        indicies[pos] = i

    alpha = forward_algorithm(model, validation_corpus)
    return alpha[len(validation_corpus.split()) - 1][indicies['+']]


def bayesian_information_criterion(model : dict, validation_corpus : str, likelihood : float) -> float:
    n = len(validation_corpus.split())

    # Find number of model parameters (emissions + transitions)
    k = 0
    for node in model.values():
        k += len(node.emissions)
        k += len(node.transitions)

    return k * np.log(n) - 2 * np.log(likelihood)



def train_with_greedy(model : dict, lemmatized_corpus : list, pos_corpus : list) -> dict:

    # NOTE:
    # We use a tiny set for the validation set
    # This is because we must re-compute the forward algorithm for each edge in a fully connected graph
    # Given n nodes, that is n^2 times to compute the forward algorithm 
    # This repeats for as long as the BIC continues improving
    # It would take too much time to train the model so we decided to use a small validation set

    # Map the model's POS to their index in the cache
    indicies = {}
    for i, pos in enumerate(model.keys()):
        indicies[pos] = i

    # Create a validation set to evaluate the model
    train_ratio = int(len(lemmatized_corpus) * 0.99999)
    train_corpus = ' '.join(lemmatized_corpus[:train_ratio])
    train_corpus_pos = ' '.join(pos_corpus[:train_ratio])
    validation_corpus = ' '.join(lemmatized_corpus[train_ratio:])

    # Add start and end padding
    train_corpus = train_corpus.strip()
    train_corpus_pos = train_corpus_pos.strip()
    if train_corpus[0] != '+':
        train_corpus = '+ ' + train_corpus
        train_corpus_pos = '+ ' + train_corpus_pos
    if train_corpus[-1] != '+':
        train_corpus = train_corpus + ' +'
        train_corpus_pos = train_corpus_pos + ' +'
    train_corpus = train_corpus.split()
    train_corpus_pos = train_corpus_pos.split()

    validation_corpus = validation_corpus.strip()
    if validation_corpus[0] != '+':
        validation_corpus = '+ ' + validation_corpus
    if validation_corpus[-1] != '+':
        validation_corpus = validation_corpus + ' +'
    
    # We get a fully connected graph, but we need to obtain some base probabilities for emissions and transitions for the BIC pruning
    # We will use the training set to build the MLE probabilities 
    model = train_with_mle(model, train_corpus, train_corpus_pos)


    # Consider every edge. Remove the edge that improves the BIC the most (lower is better)
    # Repeat until the BIC no longer improves
    bic_improving = True
    while (bic_improving):
        bic_improving = False

        initial_bic = bayesian_information_criterion(model, validation_corpus, model_likelihood(model, validation_corpus))

        # Add an edge that minimizes the BIC
        transition = None
        for pos, node in model.items():
            for neighbor_pos, neighbor in model.items():
                    if neighbor_pos in node.transitions:
                        # save this edge
                        saved_transition = node.transitions[neighbor_pos]
                        # remove the edge
                        del node.transitions[neighbor_pos]
                        # Recompute BIC
                        new_bic = bayesian_information_criterion(model, validation_corpus, model_likelihood(model, validation_corpus))
                        if new_bic < initial_bic or initial_bic == np.inf:
                            initial_bic = new_bic
                            transition = saved_transition
                        # Add the deleted edge back into the graph
                        node.transitions[neighbor_pos] = saved_transition
        # Check if there was an edge that minimized the bic and if so, delete it 
        if transition:
            source = transition.source
            sink = transition.sink
            del model[source].transitions[sink]
            bic_improving = True

            # normalize the transition proobabilities now that the transition has been deleted
            sum = 0
            for t in model[source].transitions.values():
                sum += t.probability
            for t in model[source].transitions.values():
                t.probability = t.probability / sum

    
    return model

    


def build_model(use_mle = True, save = False) -> dict:
    '''
        Builds an HMM model

        Parameters
        --------------

        use_mle : bool, default=True
            A flag indicating whether to build this model using MLE estimates for transition/emission or using a greedy search  

        save : bool, default=False
            A flag indicating whether or not to save the model as a .data file

        :returns: A dictionary containing the built model
    '''

    # Get the lemmatized & POS tagged training corpus
    with open('P2\\lematized_corpus.txt', 'r') as file:
        lemmatized_corpus = file.read()
        file.close()

    with open('P2\\pos_tagged_corpus.txt', 'r') as file:
        pos_corpus = file.read()
        file.close()



    # Create a train and test set for both lemmas and POS
    train_ratio = int(len(lemmatized_corpus.split()) * 0.9)
    train_corpus = ' '.join(lemmatized_corpus.split()[:train_ratio])
    train_corpus_pos = ' '.join(pos_corpus.split()[:train_ratio])
    test_corpus = ' '.join(lemmatized_corpus.split()[train_ratio:])
    test_corpus_pos = ' '.join(pos_corpus.split()[train_ratio:])

    # Add start and end padding
    train_corpus = train_corpus.strip()
    train_corpus_pos = train_corpus_pos.strip()
    if train_corpus[0] != '+':
        train_corpus = '+ ' + train_corpus
        train_corpus_pos = '+ ' + train_corpus_pos
    if train_corpus[-1] != '+':
        train_corpus = train_corpus + ' +'
        train_corpus_pos = train_corpus_pos + ' +'
    train_corpus = train_corpus.split()
    train_corpus_pos = train_corpus_pos.split()

    test_corpus = test_corpus.strip()
    test_corpus_pos = test_corpus_pos.strip()
    if test_corpus[0] != '+':
        test_corpus = '+ ' + test_corpus
        test_corpus_pos = '+ ' + test_corpus_pos
    if test_corpus[-1] != '+':
        test_corpus = test_corpus + ' +'
        test_corpus_pos = test_corpus_pos + ' +'
    test_corpus = test_corpus.split()
    test_corpus_pos = test_corpus_pos.split()
    
    # Construct the vocabulary 
    pos_vocab = defaultdict()

    for i in range (len(train_corpus)):
        if train_corpus_pos[i] != '+':
            if train_corpus_pos[i] in pos_vocab:
                pos_vocab[train_corpus_pos[i]].append(train_corpus[i])
            else:
                pos_vocab[train_corpus_pos[i]] = [train_corpus[i]]
    
    # Make sure there are no duplicates in the vocabulary
    for vocab in pos_vocab.values():
        vocab = list(set(vocab))

    # Model is a dictionary mapping (pos --> Node object)
    model = {}
    for key, value in pos_vocab.items():
        # Construct the Emission objects
        emissions = {}
        for token in value:
            if token not in emissions:
                emissions[token] = Emission(token=token)
        
        # Construct the Node objects 
        model[key] = Node(pos=key, transitions={}, emissions=emissions)

    # Add the START/END token to the graph. Only emits '+'
    model['+'] = Node(pos='+', emissions={'+' : Emission(token='+', probability=1.0)})


    # Create a fully connected graph
    for node in model.values():
        # Construct the Transition objects
        transitions = {}
        for neighbor in model.values():
            if neighbor.pos not in transitions:
                transitions[neighbor.pos] = Transition(source=node.pos, sink=neighbor.pos)
        node.transitions = transitions
    if use_mle:
        # Train the model probabilities using MLE estimates
        model = train_with_mle(model, train_corpus, train_corpus_pos)
    else:
        # Train the model probabilities using greedy BIC search
        # start with fully connected model
        # remove edges if the BIC goes down
        model = train_with_greedy(model, train_corpus, train_corpus_pos)

    if save and use_mle:
        save_model(model, "full_hmm_model")
    elif save and not use_mle:
        save_model(model, "BIC_hmm_model")

    return model


