def permutation_importance(filename, BREAST, LUNG, MELANOMA):
    with open(filename) as f:
        lines = f.readlines()
    if BREAST and LUNG and MELANOMA:
        for c1, line in enumerate(lines):
            if "Breast" in line and "Melanoma" in line and "Lung" in line:
                for c2, line in enumerate(lines[c1:], start=c1):
                    if "Permutation Importance" in line:
                        permutation_importance = permutations(lines, c2 + 1)
                        break
    elif BREAST and LUNG and not MELANOMA:
        for c1, line in enumerate(lines):
            if "Breast" in line and "Lung" in line and "Melanoma" not in line:
                for c2, line in enumerate(lines[c1:], start=c1):
                    if "Permutation Importance" in line:
                        permutation_importance = permutations(lines, c2 + 1)
                        break
    elif BREAST and MELANOMA and not LUNG:
        for c1, line in enumerate(lines):
            if "Breast" in line and "Melanoma" in line and "Lung" not in line:
                for c2, line in enumerate(lines[c1:], start=c1):
                    if "Permutation Importance" in line:
                        permutation_importance = permutations(lines, c2 + 1)
                        break
    elif LUNG and MELANOMA and not BREAST:
        for c1, line in enumerate(lines):
            if "Lung" in line and "Melanoma" in line and "Breast" not in line:
                for c2, line in enumerate(lines[c1:], start=c1):
                    if "Permutation Importance" in line:
                        permutation_importance = permutations(lines, c2 + 1)
                        break
    return permutation_importance


def permutations(lines, c):
    features_permutation = {}
    current = None
    for i in range(c, len(lines)):
        if "LIME" in lines[i]:
            break
        if "->" in lines[i]:
            if current != None:
                features_permutation[current]["mean"] /= features_permutation[current]["count"] 
                features_permutation[current]["std"] /= features_permutation[current]["count"]
            features_permutation[lines[i][:4]] = {"count": 0, "mean": 0, "std": 0}
            current = lines[i][:4]
        if "->" not in lines[i] and "(" in lines[i]:
            features_permutation[current]["count"] += 1
            features_permutation[current]["mean"] += float(lines[i].split(", ")[1])
            features_permutation[current]["std"] += float(lines[i].split(", ")[2][:-2])
    features_permutation[current]["mean"] /= features_permutation[current]["count"] 
    features_permutation[current]["std"] /= features_permutation[current]["count"]
    return features_permutation


def save_permutation_importance(filename, features_permutation, BREAST, LUNG, MELANOMA, model):
    with open(filename, "a") as f:
        f.write(model + "\n")
        if BREAST:
            f.write("Breast cancer ")
        if LUNG:
            f.write("Lung cancer ")
        if MELANOMA:
            f.write("Melanoma ")
        f.write("\n")
        for feature in features_permutation:
            f.write(feature + ": " + str(features_permutation[feature]) + "\n")
        
        
if __name__ == "__main__":
    BREAST = True
    LUNG = True
    MELANOMA = False
    permutations_result = permutation_importance("Results/features_LDA.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "LDA")
    permutations_result = permutation_importance("Results/features_logistic.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "logistic")
    permutations_result = permutation_importance("Results/features_svm.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "svm")
    permutations_result = permutation_importance("Results/features_random_forest.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "random_forest")
    BREAST = True
    LUNG = False
    MELANOMA = True
    permutations_result = permutation_importance("Results/features_LDA.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "LDA")
    permutations_result = permutation_importance("Results/features_logistic.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "logistic")
    permutations_result = permutation_importance("Results/features_svm.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "svm")
    permutations_result = permutation_importance("Results/features_random_forest.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "random_forest")
    BREAST = False
    LUNG = True
    MELANOMA = True
    permutations_result = permutation_importance("Results/features_LDA.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "LDA")
    permutations_result = permutation_importance("Results/features_logistic.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "logistic")
    permutations_result = permutation_importance("Results/features_svm.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "svm")
    permutations_result = permutation_importance("Results/features_random_forest.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "random_forest")
    BREAST = True
    LUNG = True
    MELANOMA = True
    permutations_result = permutation_importance("Results/features_LDA.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "LDA")
    permutations_result = permutation_importance("Results/features_logistic.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "logistic")
    permutations_result = permutation_importance("Results/features_svm.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "svm")
    permutations_result = permutation_importance("Results/features_random_forest.txt", BREAST, LUNG, MELANOMA)
    save_permutation_importance("Results/permutation_importance.txt", permutations_result, BREAST, LUNG, MELANOMA, "random_forest")
    
    