import pandas as pd
from sklearn.metrics import rand_score
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
import os

def main(exp1, depth1):
    array = pd.read_csv("./csv_laurent.csv")
    clfs = pd.read_pickle("./mean_vals_trees")
    clf1 = clfs[5*depth1 + exp1 - 6]

    dot_data = tree.export_graphviz(clf1, out_file=None,
                            feature_names=['Mol_weight', 'Isoelectric_point', 'GRAVY Score'],
                            filled=True, rounded=True,
                            special_characters=True, rotate=True, leaves_parallel=True)

    graph = graphviz.Source(dot_data)
    graph.dpi = 500
    graph.size = "30,30!"
    graph.render(f"Arbres_experience_mean_exp_{exp1}_depth={depth1}")
    os.remove(f"Arbres_experience_mean_exp_{exp1}_depth={depth1}")
    # plt.imread(f"./Arbres_experience_mean_exp_{exp1}_depth={depth1}.pdf")
    # plt.show()


if __name__ == "__main__":
    # exp1 = int(input("Quelle est l'expérience que tu veux effectuer ? "))
    #
    # while (exp1 < 1 or exp1 > 5):
    #     exp1 = int(input("Valeur incorrecte. Quelle est la première expérience que tu veux comparer ? "))
    #
    # depth1 = int(input("Quelle est la profondeur de ton arbre? "))
    # while depth1 <= 0:
    #     depth1 = int(input("La profondeur doit être une valeur positive. Quelle est la profondeur de ton arbre? "))
    for exp1 in [1,2,3,4,5]:
        for depth1 in [4,5,6,7]:
            main(exp1, depth1)