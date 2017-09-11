import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    @staticmethod
    def BIC(L, p, N):
        """ compute Bayesian information criteria using
        this formula:
        BIC = -2 * logL + p * logN
        Note: in this function L is already log so
        we do not use np.log()
        """

        BIC = -2 * L + p * np.log(N)

        return BIC

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # compute range of n (including both numbers)
        components_diff = self.max_n_components - self.min_n_components + 1
        range_n = np.linspace(self.min_n_components, self.max_n_components,
                              components_diff)
        range_n = [int(i) for i in range_n]

        for idx, n in enumerate(range_n):
            # estimate model for n
            try:
                model_n = self.base_model(n)
                # compute likelihood
                L = model_n.score(self.X, self.lengths)
                # number of data points and features
                N, f = self.X.shape # N - data points, f - features
                # number of parameters
                #p = n * (n-1) + 2 * self.X.shape[1] * n
                p = n**2 + 2 * n * f - 1
                # compute BIC score
                BIC = SelectorBIC.BIC(L, p, N)
            except:
                BIC = np.inf
            # for the first run, crash if first run throws an exception
            if idx == 0:
                best_BIC = BIC
                best_num_components = n
            else:
                if best_BIC < BIC:
                    best_BIC = BIC
                    best_num_components = n

        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    @staticmethod
    def DIC(hmm_model, words, this_word, hwords, logL):
        """purpose: calculate DIC
        """

        word_scores = 0
        competing_words = list(words)
        competing_words.remove(this_word)
        # sum over all words except i-th word
        for word in competing_words:
            X, lengths = hwords[word]
            word_scores += hmm_model.score(X, lengths)

        # calculate score
        DIC = logL - word_scores / len(competing_words)

        return DIC

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        components_diff = self.max_n_components - self.min_n_components + 1
        range_n = np.linspace(self.min_n_components, self.max_n_components,
                              components_diff)
        range_n = [int(i) for i in range_n]

        for idx, n in enumerate(range_n):
            # build model
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(
                                        self.X, self.lengths)
                # calculate likelihood
                logL = hmm_model.score(self.X, self.lengths)

                DIC = SelectorDIC.DIC(hmm_model, self.words, self.this_word,
                                    self.hwords, logL)
            except:
                DIC = -np.inf
            # for the first run, it will crash if first attempt
            # to build model will crash
            if idx == 0:
                best_DIC = DIC
                best_num_components = n
            else:
                if best_DIC < DIC:
                    best_DIC = DIC
                    best_num_components = n

        return self.base_model(best_num_components)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # number of CV splits
        num_cv_splits = min(3, len(self.sequences[0]))
        split_method = KFold(num_cv_splits) # 3 is the number of CV splits

        # compute range of n (including both numbers)
        components_diff = self.max_n_components - self.min_n_components + 1
        range_n = np.linspace(self.min_n_components, self.max_n_components,
            components_diff)
        range_n = [int(i) for i in range_n]

        # for each in model
        models_dict = {}
        logL_dict = {}
        for n in range_n:
            # for cross-validation folds
            logL = []
            models = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences[0]):
                try:
                    train_x, train_y = combine_sequences(cv_train_idx, self.sequences)
                    test_x, test_y = combine_sequences(cv_test_idx, self.sequences)
                    # build model
                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(
                                            train_x, train_y)
                    # calculate logL
                    logL.append(hmm_model.score(test_x, test_y))

                    # append model object
                    models.append(hmm_model)

                    # add to dictionaries
                    models_dict[n] = models
                    logL_dict[n] = np.mean(logL)
                # this is for splitting words and HMM crashing
                except:
                    pass

        # select the best model by choosing max mean likelihood
        if len(logL_dict) > 0:
            best_num_components = max(logL_dict, key=logL_dict.get)
        else:
            best_num_components = self.n_constant

        return self.base_model(best_num_components)
