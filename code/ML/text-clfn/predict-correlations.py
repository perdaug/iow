
"""
VERSION
- Python 3

FUNCTION
- Classifying the images based on textual features.
"""

import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np
from optparse import OptionParser
import pickle as pkl


op = OptionParser()
op.add_option('--source', action='store', type=str,
              help='The source of the corpora.')
op.add_option('--lexicon', action='store', type=str,
              help='The source of the lexicon.')
op.add_option('--claim', action='store', type=str,
              help='The choice of lexicon.')
op.add_option('--experiment', action='store', type=str,
              help='The choice of an experiment.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_LEXICON = PATH_HOME + '/data/FE/textual/' + opts.lexicon \
    + '/features-separate_' + opts.claim + '/'
PATH_SOURCE = PATH_HOME + '/data/FE/textual/' + opts.source
PATH_OUT = PATH_HOME + '/data/ML/text-clfn/' + opts.claim + '/' \
    + opts.source + '/' + opts.lexicon + '/'
PATH_EXPERIMENT = '{}/data/settings/dirs_{}-{}.txt'.format(
    PATH_HOME, opts.claim, opts.experiment)
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def main():
    vectoriser_count = CountVectorizer(stop_words='english')
    transformer_tfidf = TfidfTransformer()
    '''
    Initialising the lexicon.
    '''
    file_text = open(PATH_EXPERIMENT, 'r')
    categories = file_text.read().split('\n')
    lexicon_sk = load_files(PATH_LEXICON, categories=categories)
    # print(lexicon_sk.data[0])
    lexicon_vectorised = vectoriser_count.fit_transform(lexicon_sk.data)
    X_lexicon = transformer_tfidf.fit_transform(lexicon_vectorised)
    y_lexicon = lexicon_sk.target
    # print(len(y_lexicon))
    # return
    # print(lexicon_sk.target_names)
    # print(lexicon_sk.target)
    # names_target = np.array(lexicon_sk.target_names)[lexicon_sk.target]
    names_target = np.array(lexicon_sk.target_names)
    print(lexicon_sk.target_names)
    # print(names_target )
    pkl.dump(names_target, open(PATH_OUT + 'target.pkl', 'wb'))
    # return

    '''
    Initialising the corpora.
    '''
    corpora_sk = load_files(PATH_SOURCE, categories=categories)
    corpora_vectorised = vectoriser_count.transform(corpora_sk.data)
    X_corpora = transformer_tfidf.transform(corpora_vectorised)
    y_true_corpus = corpora_sk.target
    '''
    Running the parameter tuning
    '''
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import svm
    from sklearn import linear_model
    # from sklearn.grid_search import GridSearchCV
    # from sklearn.model_selection import StratifiedShuffleSplit
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    # range_C = np.logspace(-6, 6, 3)
    # tuned_parameters = [{'C': range_C}]
    # clf = GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, cv=2, scoring='precision_macro')

    # for (idxs_train, idxs_test) in cv.split(X_corpora, y_true_corpus):
    #     X_train = X_corpora[idxs_train]
    #     X_test = X_corpora[idxs_test]
    #     y_train = y_true_corpus[idxs_train]
    #     y_test = y_true_corpus[idxs_test]
    #     clf.fit(X_train, y_train.ravel())
    #     y_pred = clf.predict(X_test)
    #     print(clf.best_params_)
    #     report_clf = metrics.classification_report(y_test, y_pred)
    #     print(report_clf)
    # return
    '''
    Running the classification
    '''
    # clfs = [MultinomialNB(), svm.SVC(decision_function_shape='ovo'),
    #         linear_model.LogisticRegression(C=1e5)]
    # names_clf = ['nb', 'svm', 'log-reg']
    clfs = [MultinomialNB(), linear_model.LogisticRegression(C=1)]
    names_clf = ['nb', 'log-reg']
    for clf, name_clf in zip(clfs, names_clf):
        print('Running model: %s' % name_clf)
        clf.fit(X_lexicon, y_lexicon)
        y_pred_corpus = clf.predict(X_corpora)
        report_clf = metrics.classification_report(y_true_corpus,
                                                   y_pred_corpus)
        matrix_conf = metrics.confusion_matrix(y_true_corpus, y_pred_corpus)
        path_model = PATH_OUT + name_clf + '/'
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        pkl.dump(matrix_conf, open(path_model + 'matrix-conf.pkl', 'wb'))
        print(report_clf)
        print(matrix_conf)
    print(dir(clf))


if __name__ == '__main__':
    main()
