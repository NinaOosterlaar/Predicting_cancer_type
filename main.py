from classifiers import *
from interpretation import *


def main(classifier, BREAST, LUNG, MELANOMA, save_features = True):
    if classifier == "LDA":
        parameters = get_parameters("Hyperparameters/LDA.txt", BREAST, LUNG, MELANOMA)
        X, y = choose_cancers(BREAST, LUNG, MELANOMA)
        values, reports, coef = linear_discriminant_analysis(parameters, X, y)
        scores = manage_scores(values, reports, BREAST, LUNG, MELANOMA)
        save_scores(scores, "Results/score_LDA.txt", "LDA", BREAST, LUNG, MELANOMA) 
        print("LDA performed")
        if save_features:
            lime_permutation = LIME(values, X, y)
            print("Lime performed")
            permutation_features = feature_permutation(values, X, y)
            print("Permutation performed")
            filename = "Results/features_LDA.txt"
            save_features(filename, permutation_features, lime_permutation, coef, BREAST, LUNG, MELANOMA)
    if classifier == "logistic":
        parameters = get_parameters("Hyperparameters/logistic_regression.txt", BREAST, LUNG, MELANOMA)
        X, y = choose_cancers(BREAST, LUNG, MELANOMA)
        values, reports, coef = logistic_regression(parameters, X, y)
        scores = manage_scores(values, reports, BREAST, LUNG, MELANOMA)
        save_scores(scores, "Results/score_logistic.txt", "logistic", BREAST, LUNG, MELANOMA) 
        print("Logistic performed")
        if save_features:
            lime_permutation = LIME(values, X, y)
            print("Lime performed")
            permutation_features = feature_permutation(values, X, y)
            print("Permutation performed")
            filename = "Results/features_logistic.txt"
            save_features(filename, permutation_features, lime_permutation, coef, BREAST, LUNG, MELANOMA)
    if classifier == "SVM":
        parameters = get_parameters("Hyperparameters/svm.txt", BREAST, LUNG, MELANOMA)
        X, y = choose_cancers(BREAST, LUNG, MELANOMA)
        values, reports, coef = support_vector_machine(parameters, X, y)
        scores = manage_scores(values, reports, BREAST, LUNG, MELANOMA)
        save_scores(scores, "Results/score_svm.txt", "SVM", BREAST, LUNG, MELANOMA) 
        print("SVM performed")
        if save_features:
            lime_permutation = LIME(values, X, y)
            print("Lime performed")
            permutation_features = feature_permutation(values, X, y)
            print("Permutation performed")
            filename = "Results/features_svm.txt"
            save_features(filename, permutation_features, lime_permutation, coef, BREAST, LUNG, MELANOMA)
    if classifier == "random_forest":
        parameters = get_parameters("Hyperparameters/random_forest.txt", BREAST, LUNG, MELANOMA)
        X, y = choose_cancers(BREAST, LUNG, MELANOMA)
        values, reports, coef = random_forest(parameters, X, y)
        scores = manage_scores(values, reports, BREAST, LUNG, MELANOMA)
        save_scores(scores, "Results/score_random_forest.txt", "RandomForest", BREAST, LUNG, MELANOMA) 
        print("Random Forest performed")
        if save_features:
            lime_permutation = LIME(values, X, y)
            print("Lime performed")
            permutation_features = feature_permutation(values, X, y)
            print("Permutation performed")
            filename = "Results/features_random_forest.txt"
            save_features(filename, permutation_features, lime_permutation, coef, BREAST, LUNG, MELANOMA)


if __name__ == "__main__":
    classifier = "random_forest"
    BREAST = True
    LUNG = True
    MELANOMA = True
    main(classifier, BREAST, LUNG, MELANOMA, save_features=False)