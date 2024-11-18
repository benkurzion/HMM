
# HMM based unscrambler by Ben Kurzion, Rohith Eshwarwak, and Akash Rana

## Corpus used
We did not want to pay for a full dataset from *English Corpora* but still wanted a corpus with an adequate number of tokens. 
To solve this issue, we combined the Corpus of Contemporary American English (COCA) and the Corpus of Historical American English (COHA) corpuses.
The corpuses were stemmed using the *nltk* library.
For model evaluation purposes, we split the combined corpus into 70% training data and 30% testing data.

You can find our corpus saved in **https://drive.google.com/drive/folders/1aqBkDyF4yoopot87uUuAhOzjXwvIcKv5?usp=sharing**

## Models implemented
We implemented a maximum likelihood *HMM* model. The model is initialized as a fully connected graph--each part of speech can transition to every other part of speech. Emission and transition probabilities were trained using MLE on the training corpus. 

To prevent some transitions from being 0 due to a finite dataset, we used +1 smoothing. The adjusted MLE counts (prenormalization) for every transition were $count_{\text mle} + 1$

Our second model used a greedy seach. The greedy search starting state was a fully connected graph trained using the MLE approach as described above. Then, the greedy search would iterate through every transition in the graph and remove the transition which lowered the *bayesian information criterion* the most. The greedy search continued pruning edges until the *BIC* no longer improved. 

## Mapping models to files/lexicons
We have included the part of speech lexicon counts in the *hmm_lexicon.txt* file. The file contains each unique part of speech with the corresponding number of times this part of speech appears in the **full** corpus.
The lexicon is sorted by the frequency each word appears in descending order. It is formatted as *some_pos : count*

The other files we have added are pre-trained models for MLE trained and greedy search trained. They are called *full_hmm_model.pickle* and *BIC_hmm_model.pickle* respectively. To load the pickle files, please use the provided method *load_model(filepath) -> dict*

## How to evaluate
Before evaluating our model, please navigate to our data and download the files *train_corpus.txt* and *test_corpus.txt*

These are saved in **https://drive.google.com/drive/folders/1aqBkDyF4yoopot87uUuAhOzjXwvIcKv5?usp=sharing**

To evaluate our models on their perplexity and their ability to unscramble strings, please refer to the *evaluate.py* file. This file can be run from the command line.
It takes a number of arguments in the form: 

*-model n use_smoothing -evaluate file_path*

Where 
- n : int
- use_smoothing : [True, False]
- file_path : str

*-unscramble n use_smoothing file_path*

Where 
- n : int
- use_smoothing : [True, False]
- file_path : str

If you want to check a model's perplexity using a text that you provide, you would use the first option. 
For example, if you want to evaluate the perplexity of a bigram with Good-Turing smoothing on a file called *test.txt*, you would run the file like so:

**-model 2 True -evaluate test.txt**

If you wanted to unscramble a string stored in a file, you would use the second option. 
For example, if you wanted to unscrable a string stored in a file called *scrambled.txt* using a unigram with Good-Turing smoothing, you would run the file like so:

**-unscramble 1 False scrambled.txt**

From the command line, you could run the *evaluate.py* file like so:

**python evaluate.py -model 2 True -evaluate test.txt -unscramble 1 False scrambled.txt**
