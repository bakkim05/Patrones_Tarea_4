#!/usr/bin/env python

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
from matplotlib.pyplot import show
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay


def main():
    print("GETTING DATA...")
    data = get_data()

    
    #clf = SVC(probability=False, kernel="rbf", C=1.8, gamma=0.001)
    clf = SVC(probability=False, kernel="sigmoid", C=1.8, gamma=0.0073)
    #clf = SVC(probability=False, kernel="linear", C=1.8, gamma=0.0073)

    print("FITTING DATA...")
    
    examples = len(data["train"]["X"])
    clf.fit(data["train"]["X"][:examples], data["train"]["y"][:examples])
    dump(clf, 'sigmoid-073-28.joblib')
    #clf = load('linear-073-28.joblib')


    print("ANALYZING DATA...")
    analyze(clf, data)


def analyze(clf, data):
    # Get confusion matrix
    predicted = clf.predict(data["test"]["X"])
    y_test = data["test"]["y"]
    
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(y_test, predicted))
    print("Precision: %0.4f" % metrics.precision_score(y_test, predicted,average='weighted'))
    print("Recall: %0.4f" % metrics.recall_score(y_test, predicted,average='weighted'))
    print("F1: %0.4f" % metrics.f1_score(y_test, predicted,average='weighted'))

    #Confusion Matrix
    cm = confusion_matrix(data["test"]["y"], predicted)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()
    show()
    

    #ROC Curve
 #   fpr, tpr, _ = roc_curve(y_test, predicted)
 #   roc_display = RocCurveDisplay(fpr, tpr)
 #   roc_display.plot()
 #   show()

    #Precision Recall
##    prec, recall, _ = precision_recall_curve(y_test, predicted, pos_label = clf.classes_[1])
##    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
##    pr_display.plot()
##    show()



##    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,8))
##    cm_display.plot(ax=ax1)
##    roc_display.plot(ax=ax2)
##    pr_display.plot(ax=ax3)



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
