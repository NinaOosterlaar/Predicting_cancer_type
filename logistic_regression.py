from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from create_data import choose_cancers
import warnings
from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)

BREAST = True
LUNG = True
TESTICULAR = False
MELANOMA = False
LIVER = False


def parameters_grid():
    """Defines the parameters for the grid search."""
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga", "newton-cholesky"]
    tol = [1e-4, 1e-3, 1e-2]
    penalty = ["l1", "l2", "elasticnet", None]
    class_weight = ["balanced", None]
    return {"C": C, "solver": solver, "tol": tol, "penalty": penalty, "class_weight": class_weight}


def logistic_regression():
    """Performs a grid search for the logistic regression classifier."""
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LogisticRegression(max_iter=5000)
    parameters = parameters_grid()
    grid = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5))
    grid.fit(X, y)
    print("Logistic Regression")
    print("Best parameters: ", grid.best_params_)
    print("Best score: ", grid.best_score_)
    return grid.best_params_, grid.best_score_


def save_best_parameters(filename: str, best_params: dict, best_score: float):
    """Saves the best parameters and the best score to a file."""
    with open(filename, "a") as file:
        if BREAST:
            file.write("Breast cancer ")
        if LUNG:
            file.write("Lung cancer ")
        if MELANOMA:
            file.write("Melanoma ")
        file.write("\n")
        file.write("Best parameters: " + str(best_params) + "\n")
        file.write("Best score: " + str(best_score) + "\n")
    
    
if __name__ == "__main__":
    param, score = logistic_regression()
    save_best_parameters("Hyperparameters/logistic_regression.txt", param, score)
                    

