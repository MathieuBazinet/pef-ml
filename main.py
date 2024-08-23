from utils import experiments, mean_std_matrix
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
import seaborn as sn

def main():
    array = pd.read_csv("./csv_laurent.csv")
    test_cols = ['Mol_weight', 'Isoelectric_point', 'GRAVY Score', "Y"]

    all_res_three, three_stds, _, best_r2_clf, _, _ = experiments(array, test_cols, "three_vals")
    output_file = join(dirname(abspath(__file__)), f"three_vals_trees")
    with open(output_file, 'wb') as out_file:
        pickle.dump(best_r2_clf, out_file, protocol=4)

    all_res_mean, mean_stds, _, best_r2_clf, _, _ = experiments(array, test_cols, "mean")
    output_file = join(dirname(abspath(__file__)), f"mean_vals_trees")
    with open(output_file, 'wb') as out_file:
        pickle.dump(best_r2_clf, out_file, protocol=4)

    all_res_median, median_stds, _, best_r2_clf, _, _ = experiments(array, test_cols, "median")
    output_file = join(dirname(abspath(__file__)), f"median_vals_trees")
    with open(output_file, 'wb') as out_file:
        pickle.dump(best_r2_clf, out_file, protocol=4)

    all_res_mmit, mmit_stds, _, _, _, _ = experiments(array, test_cols, "mmit")

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20, 20))
    fig.suptitle("R^2 for all methods")
    for exp in range(5):
        X = pd.DataFrame(
            [all_res_three[exp, 0, :], all_res_mean[exp, 0, :], all_res_median[exp, 0, :], all_res_mmit[exp, 0, :]],
            index=["Three values", "mean", "median", "mmit"], columns=[f"Depth {i + 1}" for i in range(8)])
        Y = pd.DataFrame(
            [all_res_three[exp, 1, :], all_res_mean[exp, 1, :], all_res_median[exp, 1, :], all_res_mmit[exp, 1, :]])
        std_three = mean_std_matrix(all_res_three, three_stds)
        std_mean = mean_std_matrix(all_res_mean, mean_stds)
        std_median = mean_std_matrix(all_res_median, median_stds)
        std_mmit = mean_std_matrix(all_res_mmit, mmit_stds)
        Z = np.vstack((std_three[exp, 0, :], std_mean[exp, 0, :], std_median[exp, 0, :], std_mmit[exp, 0, :]))
        sn.heatmap(X, ax=ax[exp], annot=False, cmap='RdBu', vmin=0.3, vmax=0.75)
        sn.heatmap(X, ax=ax[exp], annot=Z, annot_kws={'va': 'bottom'}, fmt='', cbar=False, cmap='RdBu', vmin=0.3,
                   vmax=0.75)
        sn.heatmap(X, ax=ax[exp], annot=Y, annot_kws={'va': 'top'}, fmt='.3g', cbar=False, cmap='RdBu', vmin=0.3,
                   vmax=0.75)
        ax[exp].set_title(f"Experience {exp + 1}")
    plt.savefig("./all_heatmaps_experiments.jpg")
    plt.close()

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20, 20))
    fig.suptitle("Interval R^2 for all methods")
    for exp in range(5):
        X = pd.DataFrame(
            [all_res_three[exp, 2, :], all_res_mean[exp, 2, :], all_res_median[exp, 2, :], all_res_mmit[exp, 2, :]],
            index=["Three values", "mean", "median", "mmit"], columns=[f"Depth {i + 1}" for i in range(8)])
        Y = pd.DataFrame(
            [all_res_three[exp, 3, :], all_res_mean[exp, 3, :], all_res_median[exp, 3, :], all_res_mmit[exp, 3, :]])
        std_three = mean_std_matrix(all_res_three, three_stds)
        std_mean = mean_std_matrix(all_res_mean, mean_stds)
        std_median = mean_std_matrix(all_res_median, median_stds)
        std_mmit = mean_std_matrix(all_res_mmit, mmit_stds)
        Z = np.vstack((std_three[exp, 2, :], std_mean[exp, 2, :], std_median[exp, 2, :], std_mmit[exp, 2, :]))
        sn.heatmap(X, ax=ax[exp], annot=False, cmap='RdBu', vmin=0.3, vmax=0.75)
        sn.heatmap(X, ax=ax[exp], annot=Z, annot_kws={'va': 'bottom'}, fmt='', cbar=False, cmap='RdBu', vmin=0.3,
                   vmax=0.75)
        sn.heatmap(X, ax=ax[exp], annot=Y, annot_kws={'va': 'top'}, fmt='.3g', cbar=False, cmap='RdBu', vmin=0.3,
                   vmax=0.75)
        ax[exp].set_title(f"Experience {exp + 1}")
    plt.savefig("./all_interval_heatmaps_experiments.jpg")

if __name__ == "__main__":
    main()