# Description: This script creates the dataset for the classification task.

def retrieve_cancer(filename: str):
    """Retrieves the mutations from the chosen cancer.

    Args:
        filename (str): Filename of the cancer mutations.

    Returns:
        cancers: List of lists of mutations.
    """
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
    """Collects the mutations from the chosen cancers.

    Args:
        filenames (list): List of filenames.

    Returns:
        X: List of lists of mutations.
        y: List of labels.
    """
    X = []
    y = []
    for c, filename in enumerate(filenames):
        cancers = retrieve_cancer(filename)
        X += cancers
        y += [c] * len(cancers)
    return X, y
        
 
def choose_cancers(breast: bool = True, lung: bool = True, testicular: bool = False,
                    melanoma: bool = False, liver: bool = False):
    """Creates a dataset with the chosen cancers.

    Args:
        breast (bool, optional): Include breast cancer. Defaults to True.
        lung (bool, optional): Include lung cancer. Defaults to True.
        testicular (bool, optional): Include testicular cancer. Defaults to False.
        melanoma (bool, optional): Include melanoma. Defaults to False.
        liver (bool, optional): Include liver cancer. Defaults to False.

    Returns:
        X: List of lists of mutations.
        y: List of labels.
    """
    filenames = []
    if breast:
        filenames.append("Mutational_catalogs/BRCA_catalog_basic.txt")
    if lung:
        filenames.append("Mutational_catalogs/LUNG_catalog_basic.txt")
    if testicular:
        filenames.append("Mutational_catalogs/TGCT_catalog_basic.txt")
    if melanoma:
        filenames.append("Mutational_catalogs/SKCM_catalog_basic.txt")
    if liver:
        filenames.append("Mutational_catalogs/LIHC_catalog_basic.txt")
    
    X, y = collect_cancers(filenames)
    
    return X, y


if __name__ == "__main__":
    X, y = choose_cancers()
    