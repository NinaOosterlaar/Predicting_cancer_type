# Description: This script performs a grid search for the support vector machine classifier.

from sklearn.svm import SVC
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
    """Defines the parameters for the grid search.
    
    Returns: 
        parameters (dict): Dictionary with the parameters.
    """
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    kernel = ["linear", "rbf", "sigmoid", "poly"]
    degree = [2, 3, 4, 5]
    gamma = ["scale", "auto"]
    coef0 = [0.0, 0.1, 0.5, 1.0]
    shrinking = [True, False]
    class_weight = ["balanced", None]
    tol = [1e-4, 1e-3, 1e-2]
    return {"C": C, "kernel": kernel, "degree": degree, "gamma": gamma, "coef0": coef0, "tol": tol, "shrinking": shrinking, "class_weight": class_weight}


def support_vector_machine():
    """Performs a grid search for the support vector machine classifier.
    prints the best parameters and the best score.
    
    Returns:
        grid.best_params_ (dict): Best parameters.
        grid.best_score_ (float): Best score.
    """
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = SVC()
    parameters = parameters_grid()
    grid = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5), verbose=3)
    grid.fit(X, y)
    print("Support Vector Machine")
    print("Best parameters: ", grid.best_params_)
    print("Best score: ", grid.best_score_)
    return grid.best_params_, grid.best_score_


def save_best_parameters(filename: str, best_params: dict, best_score: float):
    """Saves the best parameters and the best score to a file.
    
    Args:
        filename (str): Filename of the output file.
        best_params (dict): Best parameters.
        best_score (float): Best score.
    """
    with open(filename, "a") as file:
        if BREAST:
            file.write("Breast cancer ")
        if LUNG:
            file.write("Lung cancer ")
        if MELANOMA:
            file.write("Melanoma ")
        file.write("Support Vector Machine\n")
        file.write("Best parameters: " + str(best_params) + "\n")
        file.write("Best score: " + str(best_score) + "\n\n")
    
    
if __name__ == "__main__":
    param, score = support_vector_machine()
    save_best_parameters("Hyperparameters/svm.txt", param, score)