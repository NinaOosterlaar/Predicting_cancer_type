from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from create_data import choose_cancers 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BREAST = True
LUNG = False
TESTICULAR = True
MELANOMA = False
LIVER = False

def random_forest():
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=5)
    print("Random Forest Classifier")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def logistic_regression():
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Logistic Regression")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def support_vector_machine():
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = SVC()
    scores = cross_val_score(clf, X, y, cv=5)
    print("Support Vector Machine")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
def linear_discriminant_analysis():
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LinearDiscriminantAnalysis()
    scores = cross_val_score(clf, X, y, cv=5)
    print("Linear Discriminant Analysis")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



if __name__ == "__main__":
    random_forest()
    logistic_regression()
    support_vector_machine()
    linear_discriminant_analysis()