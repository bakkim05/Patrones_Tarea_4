#!/usr/bin/env python

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib.pyplot import show, imshow, cm
from sklearn.model_selection import train_test_split


def main():
    print("GETTING DATA...")
    data = get_data()
    
    clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=0.0073)
    #clf = SVC(probability=False, kernel="sigmoid", C=2.8, gamma=0.0073)
    #clf = SVC(probability=False, kernel="linear", C=2.8, gamma=0.0073)

    print("PROCESSING...")
    
    examples = len(data["train"]["X"])
    clf.fit(data["train"]["X"][:examples], data["train"]["y"][:examples])

    analyze(clf, data)


def analyze(clf, data):
    # Get confusion matrix

    predicted = clf.predict(data["test"]["X"])
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(data["test"]["y"], predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data["test"]["y"], predicted))
    print("Precision: %0.4f" % metrics.precision_score(data['test']['y'], predicted,average='weighted'))
    print("Recall: %0.4f" % metrics.recall_score(data['test']['y'], predicted,average='weighted'))
    print("F1: %0.4f" % metrics.f1_score(data['test']['y'], predicted,average='weighted'))



def get_data():


    #trae datos esc
    mnist = fetch_openml("mnist_784")

    x = mnist.data
    y = mnist.target

    # Scale data to [-1, 1] - This is of mayor importance!!!
    x = x / 255.0 * 2 - 1

    x, y = shuffle(x, y, random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    data = {
        "train": {"X": x_train, "y": y_train},
        "test": {"X": x_test, "y": y_test},
    }
    
    return data


if __name__ == "__main__":
    main()
