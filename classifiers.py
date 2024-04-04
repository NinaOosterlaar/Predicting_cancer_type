from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from create_data import choose_cancers 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from read_parameters import retrieve_parameters, preprocess_parameters
from sklearn.metrics import classification_report



def random_forest(parameters, X, y):
    clf = RandomForestClassifier(**parameters)
    scores = evaluate(clf, X, y)
    return scores
    

def logistic_regression(parameters, X, y):
    clf = LogisticRegression(max_iter=5000, **parameters)
    scores = evaluate(clf, X, y)
    return scores
    

def support_vector_machine(parameters, X, y):
    clf = SVC(**parameters, probability=True)
    scores = evaluate(clf, X, y)
    return scores
    
    
def linear_discriminant_analysis(parameters, X, y):
    clf = LinearDiscriminantAnalysis(**parameters)
    scores = evaluate(clf, X, y)
    return scores
    
    
def get_parameters(filename: str, BREAST: bool = True, LUNG: bool = True, MELANOMA: bool = False):
    parameters = retrieve_parameters(filename, BREAST, LUNG, MELANOMA)
    parameters = preprocess_parameters(parameters)
    return parameters


def evaluate(clf, X, y):
    if len(set(y)) > 2:
        score = ["accuracy", "roc_auc_ovr"]
    else:
        score = ["accuracy", "roc_auc"]
    values = cross_validate(clf, X, y, scoring=score, cv=5, return_estimator=True, return_indices=True)
    reports = []
    coef = []
    for c, estimator in enumerate(values['estimator']):
        x_test = [X[i] for i in values['indices']['test'][c]]
        y_test = [y[i] for i in values['indices']['test'][c]]
        y_pred = estimator.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)
        if hasattr(estimator, "coef_"):
            coef.append(estimator.coef_)
    return values, reports, coef


def manage_scores(values, reports, coef):
    scores = {}
    for key in values.keys():
        if key != "estimator" and key != 'indices':
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


def conversion_labels(BREAST: bool = True, LUNG: bool = True, MELANOMA: bool = False):
    if BREAST and LUNG and MELANOMA:
        return {"0": "Breast cancer", "1": "Lung cancer", "2": "Melanoma"}
    elif BREAST and LUNG:
        return {"0": "Breast cancer", "1": "Lung cancer"}
    elif BREAST and MELANOMA:
        return {"0": "Breast cancer", "1": "Melanoma"}
    else:
        return {"0": "Lung cancer", "1": "Melanoma"}
    

def save_scores(scores, filename, model, BREAST: bool = True, LUNG: bool = True, MELANOMA: bool = False):
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
    BREAST = True
    LUNG = True
    MELANOMA = True
    parameters = get_parameters("Hyperparameters/svm.txt", BREAST, LUNG, MELANOMA)
    X, y = choose_cancers(BREAST, LUNG, MELANOMA)
    values, reports, coef = support_vector_machine(parameters, X, y)
    scores = manage_scores(values, reports, coef)
    save_scores(scores, "Results/score_svm.txt", "svm trial", BREAST, LUNG, MELANOMA)