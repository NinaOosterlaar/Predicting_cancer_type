from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def retrieve_cancer(filename: str):
    with open(filename, "r") as file:
        data = file.readlines()
        cancer = []
        cancers = []
        for line in data[1:]:
            if "->" not in line:
                cancers.append(cancer)
                cancer = []
            else:
                line = line.strip().split("\t")
                cancer.append(int(line[1]))
    cancers.append(cancer)
    return cancers


def collect_cancers(filenames: list):
    X = []
    for c, filename in filenames:
        cancers = retrieve_cancer(filename)
        X.append(cancers)
        y = [c] * len(cancers)
        
    




if __name__ == "__main__":
    filename = "Mutational_catalogs/BRCA_catalog_basic.txt"
    X = retrieve_cancer(filename)
    print(X)