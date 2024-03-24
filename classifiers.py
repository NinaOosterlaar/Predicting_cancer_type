from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from create_data import choose_cancers 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from read_parameters import retrieve_parameters, preprocess_parameters

BREAST = True
LUNG = True
TESTICULAR = False
MELANOMA = False
LIVER = False
        

def random_forest(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = RandomForestClassifier(**parameters)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Random Forest Classifier")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def logistic_regression(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LogisticRegression(max_iter=5000, **parameters)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Logistic Regression")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def support_vector_machine(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = SVC(**parameters)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Support Vector Machine")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
def linear_discriminant_analysis(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LinearDiscriminantAnalysis(**parameters)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Linear Discriminant Analysis")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
def get_parameters(filename: str):
    parameters = retrieve_parameters(filename, BREAST, LUNG, MELANOMA)
    parameters = preprocess_parameters(parameters)
    return parameters


if __name__ == "__main__":
    parameters = get_parameters("Hyperparameters/logistic_regression.txt")
    logistic_regression(parameters)