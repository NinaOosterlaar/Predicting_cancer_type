from classifiers import *
import lime
import lime.lime_tabular    
import numpy as np
from sklearn.inspection import permutation_importance

BREAST = True
LUNG = False
TESTICULAR = False
MELANOMA = True
LIVER = False 

features = np.array(["A->C", "A->G", "A->T", "C->A", "C->G", "C->T", "G->A", "G->C", "G->T", "T->A", "T->C", "T->G"])

def LIME(values, X, y):
    for count in range(5):
        model, x_train, y_train = process_data(values, X, y, count)
        x_train = np.array(x_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, mode="classification", feature_names=features)
        for i in range(10):
            exp = explainer.explain_instance(x_train[i], model.predict_proba, num_features=5)
            exp.show_in_notebook(show_table=True)
            print("True class: ", y_train[i])
            print("Predicted class: ", model.predict(x_train[i].reshape(1, -1)))
            print("Feature Importance:")
            for feature, weight in exp.as_list():
                print(f"{feature}: {weight}")
            print("\n")
            
            
def feature_permutation(values, X, y):
    features_permutation = {}
    for i in range(5):
        model, x_test, y_test = process_data(values, X, y, i, train = False)
        permutations = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0)

        for j in permutations.importances_mean.argsort()[::-1]:
            if permutations.importances_mean[j] - 2 * permutations.importances_std[j] > 0:
                if features[j] not in features_permutation:
                    print(features[j])
                    features_permutation[features[j]] = []
                features_permutation[features[j]].append((i, permutations.importances_mean[j], permutations.importances_std[j]))
    return features_permutation

    
    
def process_data(values, X, y, count, train = True):
    model = values['estimator'][count]
    if train:
        x_train = [X[i] for i in values['indices']['train'][count]]
        y_train = [y[i] for i in values['indices']['train'][count]]
        return model, x_train, y_train
    else:
        x_test = [X[i] for i in values['indices']['test'][count]]
        y_test = [y[i] for i in values['indices']['test'][count]]
        return model, x_test, y_test


def get_models(classifier, BREAST, LUNG, MELANOMA):
    if classifier == "RandomForest":
        parameters = get_parameters("Hyperparameters/random_forest.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = random_forest(parameters, BREAST, LUNG, MELANOMA)
        filename = "Results/features_random_forest.txt"
    elif classifier == "SVM":
        parameters = get_parameters("Hyperparameters/svm.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = support_vector_machine(parameters, BREAST, LUNG, MELANOMA)
        filename = "Results/features_svm.txt"
    elif classifier == "LDA":
        parameters = get_parameters("Hyperparameters/LDA.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = linear_discriminant_analysis(parameters, BREAST, LUNG, MELANOMA)
        filename = "Results/features_LDA.txt"
    else:
        parameters = get_parameters("Hyperparameters/logistic_regression.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = logistic_regression(parameters, BREAST, LUNG, MELANOMA)
        filename = "Results/features_logistic.txt"
    return values, reports, coef, filename


def save_features(filename, features_permutation, coef, BREAST, LUNG, MELANOMA):
    with open(filename, "a") as f:
        if BREAST:
            f.write("Breast cancer ")
        if LUNG:
            f.write("Lung cancer ")
        if MELANOMA:
            f.write("Melanoma ")
        f.write("\n")
        if coef is not None:
            f.write("Coeffecients: \n")
            for i in range(len(coef)):
                f.write("Model " + str(i) + ": \n")
                f.write(str(coef[i]) + "\n")
        f.write("Permutation Importance: \n")
        for feature in features_permutation:
            f.write(feature + ": \n")
            for value in features_permutation[feature]:
                f.write(str(value) + "\n")
            f.write("\n")
        
    

if __name__ == "__main__":
    BREAST = True
    LUNG = True
    MELANOMA = False
    values, reports, coef, filename = get_models("LDA", BREAST, LUNG, MELANOMA)
    X, y = choose_cancers(BREAST, LUNG, MELANOMA)
    # LIME(values, X, y)
    permutation_features = feature_permutation(values, X, y)
    save_features(filename, permutation_features, coef, BREAST, LUNG, MELANOMA)