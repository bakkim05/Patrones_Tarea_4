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

    for x in range(11):
      print("\nLOADING MODEL...")
      loadmodel = savedmodel(x)
      print(loadmodel)
      clf = load(loadmodel)
      print("\nANALYZING DATA...")
      analyze(clf, data)



def savedmodel(i):
    switcher={
	0:'rbf-001-28.joblib',
	1:'rbf-001-18.joblib',
	2:'sigmoid-073-28.joblib',
	3:'sigmoid-073-18.joblib',
	4:'sigmoid-001-28.joblib',
	5:'sigmoid-001-18.joblib',
    }
    return switcher.get(i,"Invalido")

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
