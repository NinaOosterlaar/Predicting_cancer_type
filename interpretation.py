from classifiers import *
import lime
import lime.lime_tabular    
import numpy as np
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

BREAST = True
LUNG = False
TESTICULAR = False
MELANOMA = True
LIVER = False 

features = np.array(["A->C", "A->G", "A->T", "C->A", "C->G", "C->T", "G->A", "G->C", "G->T", "T->A", "T->C", "T->G"])

def LIME(values, X, y, multi_class=False):
    if multi_class:
        feature_importance = {0: {}, 1: {}, 2: {}}  # Dictionary to store feature importance values
    else:
        feature_importance = {0: {}, 1: {}}

    for count in range(5):
        model, x_test, y_test = process_data(values, X, y, count)
        x_test = np.array(x_test)
        explainer = lime.lime_tabular.LimeTabularExplainer(x_test, mode="classification", feature_names=features)
        
        highest_weight = 0
        for i in range(len(x_test)):
            exp = explainer.explain_instance(x_test[i], model.predict_proba, num_features=12)  
            
            # Accumulate feature importance values
            for feature, weight in exp.as_list():
                if abs(weight) > highest_weight:
                    highest_weight = abs(weight)
                if "G->C" in feature or "G->A" in feature or "C->T" in feature:
                    if weight > 0:
                        c = 1
                    else:
                        c = 0
                    if feature not in feature_importance[c]:
                        feature_importance[c][feature] = []
                    feature_importance[c][feature].append(weight)
        print(highest_weight)
    
    # Compute average importance values for each feature
    avg_importance = {0: {feature: (np.mean(values), np.std(values), len(values)) for feature, values in feature_importance[0].items()},
                      1: {feature: (np.mean(values), np.std(values), len(values)) for feature, values in feature_importance[1].items()}}
    # avg_importance = {feature: (np.mean(values), np.std(values), len(values)) for feature, values in feature_importance.items()}
    print(avg_importance[0])
    print("\n")
    print(avg_importance[1])

    return avg_importance
            
            
def feature_permutation(values, X, y): 
    features_permutation = {}
    for i in range(5):
        model, x_test, y_test = process_data(values, X, y, i)
        permutations = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0)

        for j in permutations.importances_mean.argsort()[::-1]:
            if permutations.importances_mean[j] - 2 * permutations.importances_std[j] > 0:
                if features[j] not in features_permutation:
                    features_permutation[features[j]] = []
                features_permutation[features[j]].append((i, permutations.importances_mean[j], permutations.importances_std[j]))
    return features_permutation

    
    
def process_data(values, X, y, count):
    model = values['estimator'][count]
    x_test = [X[i] for i in values['indices']['test'][count]]
    y_test = [y[i] for i in values['indices']['test'][count]]
    return model, x_test, y_test


def get_models(classifier, X, y, BREAST, LUNG, MELANOMA):
    if classifier == "RandomForest":
        parameters = get_parameters("Hyperparameters/random_forest.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = random_forest(parameters, X, y)
        filename = "Results/features_random_forest.txt"
    elif classifier == "SVM":
        parameters = get_parameters("Hyperparameters/svm.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = support_vector_machine(parameters, X, y)
        filename = "Results/features_svm.txt"
    elif classifier == "LDA":
        parameters = get_parameters("Hyperparameters/LDA.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = linear_discriminant_analysis(parameters, X, y)
        filename = "Results/features_LDA.txt"
    else:
        print("logistic")
        parameters = get_parameters("Hyperparameters/logistic_regression.txt", BREAST, LUNG, MELANOMA)
        values, reports, coef = logistic_regression(parameters, X, y)
        filename = "Results/features_logistic.txt"
    return values, reports, coef, filename


def save_features(filename, features_permutation, features_lime, coef, BREAST, LUNG, MELANOMA):
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
        f.write("LIME: \n")
        for feature in features_lime:
            f.write(feature + ": ")
            for value in features_lime[feature]:
                f.write(str(value) + " ")
            f.write("\n")
        
    

if __name__ == "__main__":
    BREAST = True
    LUNG = False
    MELANOMA = True
    X, y = choose_cancers(BREAST, LUNG, MELANOMA)
    values, reports, coef, filename = get_models("LDA", X, y, BREAST, LUNG, MELANOMA)
    print("Models retrieved")
    lime_permutation = LIME(values, X, y, multi_class=True)
    print("Lime performed")
    # permutation_features = feature_permutation(values, X, y)
    # print("Permutation performed")
    # save_features(filename, permutation_features, lime_permutation, coef, BREAST, LUNG, MELANOMA)