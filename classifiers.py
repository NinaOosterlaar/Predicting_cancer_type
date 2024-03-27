from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from create_data import choose_cancers 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from read_parameters import retrieve_parameters, preprocess_parameters
from sklearn.metrics import classification_report

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
        score = ["accuracy", "roc_auc_ovr"]
    else:
        score = ["accuracy", "roc_auc"]
    values = cross_validate(clf, X, y, scoring=score, cv=5, return_estimator=True)
    reports = []
    for estimator in values['estimator']:
        y_pred = estimator.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        reports.append(report)
    return values, reports


def manage_scores(values, reports):
    scores = {}
    for key in values.keys():
        if key != "estimator":
            scores[key] = values[key].mean()
    labels = conversion_labels()
    for report in reports:
        for key in report.keys():
            if key in labels:
                if labels[key] not in scores:
                    scores[labels[key]] = {}
                for metric in report[key].keys():
                    if metric not in scores[labels[key]]:
                        scores[labels[key]][metric] = 0
                    scores[labels[key]][metric] += report[key][metric]
    for key in scores.keys():
        if type(scores[key]) == dict:
            for metric in scores[key].keys():
                scores[key][metric] /= len(reports)
    return scores


def conversion_labels():
    if BREAST and LUNG and MELANOMA:
        return {"0": "Breast cancer", "1": "Lung cancer", "2": "Melanoma"}
    elif BREAST and LUNG:
        return {"0": "Breast cancer", "1": "Lung cancer"}
    elif BREAST and MELANOMA:
        return {"0": "Breast cancer", "1": "Melanoma"}
    else:
        return {"0": "Lung cancer", "1": "Melanoma"}
    

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
    parameters = get_parameters("Hyperparameters/logistic_regression.txt")
    values, reports = logistic_regression(parameters)
    scores = manage_scores(values, reports)
    save_scores(scores, "Score/score_logistic.txt", "Logistic Regression")