from utils import *
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sn

# JP's
def main(exp_vals):
    array = pd.read_csv("./csv_laurent.csv")
    test_cols = ['Mol_weight', 'Isoelectric_point', 'GRAVY Score', 'L']

    all_res, standard_devs, leaf_nums, best_r2_clf, best_int_clf, train_scores = experiments(array, test_cols, exp_vals)

    plt.title("Comparison of the number of leaves for each tree")
    sn.heatmap(pd.DataFrame(leaf_nums, index=[f"Depth {i}" for i in range(1, 9)],
                            columns=[f"Exp {i}" for i in range(1, 6)]), annot=True, fmt='.3g')
    plt.savefig("./leaves_number_mmit_values.jpg")
    plt.close()

    X = pd.DataFrame(train_scores[:, 0, :].T, index=["exp1", "exp2", "exp3", "exp4", "exp5"],
                     columns=[1, 2, 3, 4, 5, 6, 7, 8])
    plt.figure(figsize=(20, 10))

    plt.title("Comparison of R^2 and RMSE on the training set for each experiments")
    sn.heatmap(X, annot=True, cmap='RdBu', vmin=0.2, vmax=0.9, fmt='.3g')
    plt.savefig("./training_scores_mmit_values.jpg")
    plt.close()

    X = pd.DataFrame(all_res[:, 0, :], index=["exp1", "exp2", "exp3", "exp4", "exp5"],
                     columns=[1, 2, 3, 4, 5, 6, 7, 8])
    std_matrix = mean_std_matrix(all_res, standard_devs)
    Y = all_res[:, 1, :]
    plt.figure(figsize=(20, 10))
    plt.title("Comparison of R^2 and RMSE for each experiments at each depth")
    sn.heatmap(X, annot=False, cmap='RdBu', vmin=0.2, vmax=0.9)
    sn.heatmap(X, annot=std_matrix[:, 0, :], annot_kws={'va': 'bottom'}, cbar=False, fmt='', cmap='RdBu')
    sn.heatmap(X, annot=Y, annot_kws={'va': 'top'}, fmt='.3g', cbar=False, cmap='RdBu')
    plt.savefig("./mmit_values_heatmaps.jpg")
    plt.close()

    X = pd.DataFrame(all_res[:, 2, :], index=["exp1", "exp2", "exp3", "exp4", "exp5"],
                     columns=[1, 2, 3, 4, 5, 6, 7, 8])
    std_matrix = mean_std_matrix(all_res, standard_devs)
    Y = all_res[:, 3, :]
    plt.figure(figsize=(15, 10))
    plt.title("Comparison of interval R^2 and RMSE for each experiments at each depth")
    sn.heatmap(X, annot=False, cmap='RdBu', vmin=0.2, vmax=0.9)
    sn.heatmap(X, annot=std_matrix[:, 2, :], annot_kws={'va': 'bottom'}, fmt='', cbar=False, cmap='RdBu', vmin=0.2,
               vmax=0.9)
    sn.heatmap(X, annot=Y, annot_kws={'va': 'top'}, fmt='.3g', cbar=False, cmap='RdBu', vmin=0.2, vmax=0.9)
    plt.savefig("./mmit_values_interval_heatmaps.jpg")

if __name__ == "__main__":
    exp_vals = "mmit"
    main(exp_vals)