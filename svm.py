#!/usr/bin/env python

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib.pyplot import show, imshow, cm
from sklearn.cross_validation import train_test_split


def main():
    """Orchestrate the retrival of data, training and testing."""
    data = get_data()

    # Get classifier

    clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=0.0073)  # cache_size=200,

    print("PROCESANDO DATOS. PORFAVOR ESPERE")

    # take all of it - make that number lower for experiments
    examples = len(data["train"]["X"])
    clf.fit(data["train"]["X"][:examples], data["train"]["y"][:examples])

    analyze(clf, data)


def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix


    predicted = clf.predict(data["test"]["X"])
    print(
        "Confusion matrix:\n%s" % metrics.confusion_matrix(data["test"]["y"], predicted)
    )
    print("Accuracy: %0.4f" % metrics.accuracy_score(data["test"]["y"], predicted))

    # Print example
    try_id = 1
    out = clf.predict(data["test"]["X"][try_id])  # clf.predict_proba
    print("out: %s" % out)
    size = int(len(data["test"]["X"][try_id]) ** (0.5))
    view_image(
        data["test"]["X"][try_id].reshape((size, size)), data["test"]["y"][try_id]
    )


def view_image(image, label=""):
    """
    View a single image.

    Parameters
    ----------
    image : numpy array
        Make sure this is of the shape you want.
    label : str
    """
    

    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """


    mnist = fetch_mldata("MNIST original")

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
