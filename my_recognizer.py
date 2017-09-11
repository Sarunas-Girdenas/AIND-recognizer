import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for X, lengths in test_set.get_all_Xlengths().values():
        # create dict for probabilities
        dict_for_prob = {}
        best_logL = None
        # get model
        for idx, (word, hmm_model) in enumerate(models.items()):
            if hmm_model:
                try:
                    # compute loglikelihood
                    logL = hmm_model.score(X, lengths)
                    dict_for_prob[word] = logL
                    # find the best within this word
                    if not best_logL:
                        best_logL = logL
                        best_word = word
                    else:
                        if logL > best_logL:
                            best_logL = logL
                            best_word = word
                except ValueError:
                    pass

        probabilities.append(dict_for_prob)
        guesses.append(best_word)

    return probabilities, guesses
