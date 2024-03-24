from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from create_data import choose_cancers
import warnings
from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)

BREAST = True
LUNG = True
TESTICULAR = False
MELANOMA = True
LIVER = False


def parameters_grid():
    """Defines the parameters for the grid search."""
    solver = ["svd", "lsqr", "eigen"]
    shrinkage = [None, "auto"]
    tol = [1e-4, 1e-3, 1e-2]
    return {"solver": solver, "shrinkage": shrinkage, "tol": tol}
    
    
def linear_discriminant_analysis():
    """Performs a grid search for the linear discriminant analysis classifier."""
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = LinearDiscriminantAnalysis()
    parameters = parameters_grid()
    grid = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5))
    grid.fit(X, y)
    print("Linear Discriminant Analysis")
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
        file.write("Logistic Regression\n")
        file.write("Best parameters: " + str(best_params) + "\n")
        file.write("Best score: " + str(best_score) + "\n\n")


if __name__ == "__main__":
    param, score = linear_discriminant_analysis()
    save_best_parameters("Hyperparameters/LDA.txt", param, score)