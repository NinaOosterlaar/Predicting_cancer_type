from sklearn.ensemble import RandomForestClassifier
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
    n_estimators = [10, 50, 100, 200, 500, 1000, 2000]
    criterion = ["gini", "entropy", "log_loss"]
    max_depth = [None, 10, 20, 30, 40, 50]
    min_samples_split = [2, 5, 10, 15]
    min_samples_leaf = [1, 2, 4, 8]
    max_features = [None, "sqrt", "log2"]
    max_leaf_nodes = [None, 10, 20, 30, 40, 50]
    bootstrap = [True, False]
    return {"n_estimators": n_estimators, "criterion": criterion, "max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "max_features": max_features, "max_leaf_nodes": max_leaf_nodes, "bootstrap": bootstrap}


def random_forest():
    """Performs a grid search for the random forest classifier."""
    X, y = choose_cancers(BREAST, LUNG, TESTICULAR, MELANOMA, LIVER)
    clf = RandomForestClassifier()
    parameters = parameters_grid()
    grid = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5), verbose=3)
    grid.fit(X, y)
    print("Random Forest Classifier")
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
        file.write("Random Forest Classifier\n")
        file.write("Best parameters: " + str(best_params) + "\n")
        file.write("Best score: " + str(best_score) + "\n\n")
        
        
if __name__ == "__main__":
    param, score = random_forest()
    save_best_parameters("Hyperparameters/random_forest.txt", param, score)