import pandas as pd
from utils import separate_peptides
from sklearn.metrics import rand_score

def main(exp1, depth1, exp2, depth2):
    array = pd.read_csv("./csv_laurent.csv")
    clfs = pd.read_pickle("./mean_vals_trees")
    clf1 = clfs[5*depth1 + exp1 - 6]
    tree1 = clf1.tree_
    dict1 = separate_peptides(array, tree1)

    liste = list(dict1.keys())
    new_dict1 = {}
    for k in liste:
        for p in dict1[k]:
            new_dict1[p] = k

    clf2 = clfs[5*depth2 + exp2 - 6]
    tree2 = clf2.tree_
    dict2 = separate_peptides(array, tree2)

    liste = list(dict2.keys())
    new_dict2 = {}
    for k in liste:
        for p in dict2[k]:
            new_dict2[p] = k

    cluster1 = []
    cluster2 = []
    for i in range(99):
        cluster1.append(new_dict1[i])
        cluster2.append(new_dict2[i])
    print("Experiment : ", exp1, " Depth : ", depth1, "Experiment : ", exp2,
          " Depth : ", depth2, "Index : ",  rand_score(cluster1, cluster2))


if __name__ == "__main__":
    exp1 = int(input("Quelle est l'expérience que tu veux effectuer ? "))

    while (exp1 < 1 or exp1 > 5):
        exp1 = int(input("Valeur incorrecte. Quelle est la première expérience que tu veux comparer ? "))

    depth1 = int(input("Quelle est la profondeur de ton arbre? "))
    while depth1 <= 0:
        depth1 = int(input("La profondeur doit être une valeur positive. Quelle est la profondeur de ton arbre? "))

    exp2 = int(input("Quelle est l'expérience que tu veux effectuer ? "))

    while (exp2 < 1 or exp2 > 5):
        exp2 = int(input("Valeur incorrecte. Quelle est la première expérience que tu veux comparer ? "))

    depth2 = int(input("Quelle est la profondeur de ton arbre? "))
    while depth2 <= 0:
        depth2 = int(input("La profondeur doit être une valeur positive. Quelle est la profondeur de ton arbre? "))

    main(exp1, depth1, exp2, depth2)