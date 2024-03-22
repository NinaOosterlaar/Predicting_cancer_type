def load_data(filename):
    with open(filename, "r") as file:
        data = file.readlines()
        genomes = {}
        for line in data[1:]:
            line = line.strip().split("\t")
            genome = line[0]
            if int(line[2]) - int(line[3]) == 0 and line[4] != "-" and line[5] != "-" and len(line[4]) == 1 and len(line[5]) == 1:
                info = {"chromosome": line[1], "location": int(line[2]), "ref": line[4], "alt": line[5]}
                if genome in genomes:
                    genomes[genome].append(info)
                else:
                    genomes[genome] = [info]
    return genomes


def create_catalog(genomes):
    catalogs = {}
    for genome in genomes:
        catalog = {"A->C": 0, "A->G": 0, "A->T": 0, "C->A": 0, "C->G": 0, "C->T": 0, "G->A": 0, "G->C": 0, "G->T": 0, "T->A": 0, "T->C": 0, "T->G": 0}
        for mutation in genomes[genome]:
            catalog[mutation["ref"] + "->" + mutation["alt"]] += 1
        catalogs[genome] = catalog
    return catalogs


def write_catalogs(catalogs, filename):
    with open(filename, "w") as file:
        for genome in catalogs:
            file.write(genome + "\n")
            for mutation in catalogs[genome]:
                file.write(mutation + "\t" + str(catalogs[genome][mutation]) + "\n")
            


if __name__ == "__main__":
    filename = "Raw_data/TGCT_mc3.txt"
    genomes = load_data(filename)
    catalogs = create_catalog(genomes)
    write_catalogs(catalogs, "Mutational_catalogs/TGCT_catalog_basic.txt")
