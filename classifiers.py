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
    scores = evaluate(clf, X, y)
    

def logistic_regression(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LogisticRegression(max_iter=5000, **parameters)
    scores = evaluate(clf, X, y)
    

def support_vector_machine(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = SVC(**parameters)
    scores = evaluate(clf, X, y)
    
    
def linear_discriminant_analysis(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LinearDiscriminantAnalysis(**parameters)
    scores = evaluate(clf, X, y)
    
    
def get_parameters(filename: str):
    parameters = retrieve_parameters(filename, BREAST, LUNG, MELANOMA)
    parameters = preprocess_parameters(parameters)
    return parameters


def evaluate(clf, X, y):
    accuracy = cross_val_score(clf, X, y, cv=5)
    roc_auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    precision = cross_val_score(clf, X, y, cv=5, scoring='precision')
    recall = cross_val_score(clf, X, y, cv=5, scoring='recall')
    specificity = cross_val_score(clf, X, y, cv=5, scoring='specificity')
    f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
    scores = {"accuracy": accuracy, "roc_auc": roc_auc, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}
    print(scores)
    return scores


def save_scores(scores, filename, model):
    with open(filename, "a") as f:
        f.write(model + "\n")


if __name__ == "__main__":
    parameters = get_parameters("Hyperparameters/logistic_regression.txt")
    logistic_regression(parameters)