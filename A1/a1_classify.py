#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os

from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility
import numpy as np

np.random.seed(401)

# classifier list
CLF_list = [(SGDClassifier(), "SGDClassifier"),
            (GaussianNB(), "GaussianNB"),
            (RandomForestClassifier(max_depth=5, n_estimators=10), "RandomForestClassifier"),
            (MLPClassifier(alpha=0.05), "MLPClassifier"),
            (AdaBoostClassifier(), "AdaBoostClassifier")]


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    ii_sum = 0
    for i in range(np.shape(C)[0]):
        ii_sum += C[i, i]
    return ii_sum / np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    lst = []
    for i in range(np.shape(C)[0]):
        lst.append(C[i, i] / sum(C[i, :]))
    return lst


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    lst = []
    for i in range(np.shape(C)[0]):
        lst.append(C[i, i] / sum(C[:, i]))
    return lst


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    print('Section 3.1')
    iBest = 0
    acc_max = 0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for clf in CLF_list:
            outf.write(f'Results for {clf[1]}:\n')  # Classifier name
            clf[0].fit(X_train, y_train)
            matrix = confusion_matrix(y_test, clf[0].predict(X_test))
            acc = accuracy(matrix)
            if acc >= acc_max:  # find the best
                acc_max = acc
                iBest = CLF_list.index(clf)
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall(matrix)]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision(matrix)]}\n')
            outf.write(f'\tConfusion Matrix: \n{matrix}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    size = [1000, 5000, 10000, 15000, 20000]
    clf, clf_name = CLF_list[iBest][0], CLF_list[iBest][1]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        for num_train in size:
            rdm = np.random.choice(np.shape(X_train)[0], size=num_train)
            if num_train == 1000:
                X_1k = X_train[rdm]
                y_1k = y_train[rdm]
            clf.fit(X_train[rdm], y_train[rdm])
            acc = accuracy(confusion_matrix(y_test, clf.predict(X_test)))
            outf.write(f'{num_train}: {acc:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    clf = CLF_list[i][0]
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        for k in [5, 50]:
            selector = SelectKBest(f_classif, k=k)
            selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_

            # for each number of features k_feat, write the p-values for
            # that number of features:
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in p_values]}\n')

        selector5 = SelectKBest(f_classif, k=5)
        x1 = selector5.fit_transform(X_1k, y_1k)
        x_test1 = selector5.transform(X_test)
        p1 = selector5.get_support(True)
        x32 = selector5.fit_transform(X_train, y_train)
        x_test32 = selector5.transform(X_test)
        p32 = selector5.get_support(True)

        inter = np.intersect1d(p1, p32)

        acc1 = accuracy(confusion_matrix(y_test, clf.fit(x1, y_1k).predict(x_test1)))
        acc32 = accuracy(confusion_matrix(y_test, clf.fit(x32, y_train).predict(x_test32)))
        outf.write(f'Accuracy for 1k: {acc1:.4f}\n')
        outf.write(f'Accuracy for full dataset: {acc32:.4f}\n')
        outf.write(f'Chosen feature intersection: {[index for index in inter]}\n')
        outf.write(f'Top-5 at higher: {[pp32 for pp32 in p32]} \n')
        outf.write(f'\nMy answers to questions go here:\n')
        outf.write(
            f'(a) 1 and 2 are "Number of first-person pronouns" and "Number of second-person pronouns." I do not think these two features can be useful since we cannot recognize	anything from person pronouns; maybe it means there are many subjective opinions.\n')
        outf.write(
            f'(b) The more data volume is, the lower the p-value is. According to the definition of p-value, the smaller p-value is, the higher the reliability of the model is. In the case of no abnormal data, the larger the data, the more natural the trained model will be. Therefore, the larger the amount of data, the smaller the p-value.\n')
        outf.write(
            f'(c) Number of first-person pronouns: First-person pronouns tend to be subjective and at the same time incite and unite people\'s emotions, like "we".\nNumber of second-person pronouns: Second-person pronouns have strong directivity, so that readers can easily substitute into the position of listeners \nNumber of third-person pronouns: Third-person pronouns can be aggressive and can be used to distinguish "enemies" like "them"\nReceptiviti_health_oriented: People pay more attention to health, so it can cause protest and some uncomfortable emotions.\nReceptiviti_intellectual: With intellectual, you tend to sound superior and have more credibility\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    print('TODO Section 3.4')
    x, y = np.vstack((X_train, X_test)), np.concatenate((y_train, y_test))
    d = {0: [], 1: [], 2: [], 3: [], 4: []}
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        kfold = KFold(n_splits=5, shuffle=True)
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        for clf_i in range(len(CLF_list)):
            kfold_accuracies = []
            for x_tra, x_test in kfold.split(x):
                CLF_list[clf_i][0].fit(x[x_tra], y[x_tra])
                matrix = confusion_matrix(y[x_test], CLF_list[clf_i][0].predict(x[x_test]))
                ac = accuracy(matrix)
                kfold_accuracies.append(ac)
                d[clf_i].append(ac)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        p_values = []
        for j in range(len(CLF_list)):
            if j == i:
                continue
            S = ttest_rel(d[j], d[i])
            p_values.append(S.pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    data = np.load(args.input)["arr_0"]
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)

    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)