from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from create_data import choose_cancers 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from read_parameters import retrieve_parameters, preprocess_parameters

BREAST = False
LUNG = True
TESTICULAR = False
MELANOMA = True
LIVER = False 

def random_forest(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = RandomForestClassifier(**parameters)
    scores = evaluate(clf, X, y)
    return scores
    

def logistic_regression(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LogisticRegression(max_iter=5000, **parameters)
    scores = evaluate(clf, X, y)
    return scores
    

def support_vector_machine(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = SVC(**parameters)
    scores = evaluate(clf, X, y)
    return scores
    
    
def linear_discriminant_analysis(parameters):
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LinearDiscriminantAnalysis(**parameters)
    scores = evaluate(clf, X, y)
    return scores
    
    
def get_parameters(filename: str):
    parameters = retrieve_parameters(filename, BREAST, LUNG, MELANOMA)
    parameters = preprocess_parameters(parameters)
    return parameters


def evaluate(clf, X, y):
    if len(set(y)) > 2:
        score = ["accuracy", "roc_auc_ovr", "precision_macro", "recall_macro", "f1_macro"]
    else:
        score = ["accuracy", "roc_auc", "precision", "recall", "f1"]
    values = cross_validate(clf, X, y, scoring=score, cv=5)
    scores = {}
    for key, value in values.items():
        scores[key] = (value.mean(), value.std())
    return scores


def save_scores(scores, filename, model):
    with open(filename, "a") as f:
        f.write(model + "\n")
        if BREAST:
            f.write("Breast cancer ")
        if LUNG:
            f.write("Lung cancer ")
        if MELANOMA:
            f.write("Melanoma ")
        f.write("\n")
        for key, value in scores.items():
            f.write(key + ": " + str(value) + "\n")


if __name__ == "__main__":
    parameters = get_parameters("Hyperparameters/LDA.txt")
    scores = linear_discriminant_analysis(parameters)
    save_scores(scores, "Score/score.txt", "LDA")