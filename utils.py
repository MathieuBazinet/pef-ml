import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn import tree
from copy import copy
import graphviz
import os
import seaborn as sn
from mmit import MaxMarginIntervalTree
from mmit.pruning import min_cost_complexity_pruning
from mmit.model_selection import GridSearchCV
from mmit.metrics import mean_squared_error as interval_mse
from mmit.metrics import zero_one_loss
from sklearn import metrics

def y_array(exp_array, experiment="mean"):
    """

    :param exp_array:
    :param experiment:
    :return: La matrice utilisée pour entraîner l'arbre, la matrice pour calculer les métriques
    de prédiction et la matrice pour les prédictions sur intervalle
    """
    exp_concat = pd.concat(exp_array, ignore_index=True, axis=1)
    y_interval = np.vstack((np.min(exp_concat, axis=1), np.max(exp_concat, axis=1))).T
    mean_exp = np.mean(pd.concat(exp_array, ignore_index=True, axis=1), axis=1)
    if experiment == "three_vals":
        return exp_concat, mean_exp, y_interval
    elif experiment == "mean":
        return mean_exp, mean_exp, y_interval
    elif experiment == "median":
        median_exp = np.median(pd.concat(exp_array, ignore_index=True, axis=1), axis=1)
        return median_exp, mean_exp, y_interval
    elif experiment == "mmit":
        return y_interval, mean_exp, y_interval
    else:
        raise("The type of experiment is not implemented.")

def test_train_split(X_array, exp, y_interval, train_index, test_index, mean_array):
    if len(exp.shape) == 1:
        train_array = X_array.iloc[train_index, :]
        test_array = X_array.iloc[test_index, :]
        y_train = exp[train_index]
        y_test = mean_array[test_index]
        y_int_test = y_interval[test_index, :]
    elif exp.shape[1] == 2:
        train_array = X_array.iloc[train_index, :]
        test_array = X_array.iloc[test_index, :]
        y_train = exp[train_index, :]
        y_test = mean_array[test_index]
        y_int_test = y_interval[test_index, :]
    elif exp.shape[1] == 3:
        train_array = pd.concat([X_array.iloc[train_index, :]] * 3, ignore_index=True, axis=0)
        test_array = pd.concat([X_array.iloc[test_index, :]], ignore_index=True, axis=0)
        y_train = pd.concat([exp.iloc[train_index, 0], exp.iloc[train_index, 1], exp.iloc[train_index, 2]],
                            ignore_index=True, axis=0)
        y_test = mean_array[test_index]
        y_int_test = y_interval[test_index, :]
    return train_array, test_array, y_train, y_test, y_int_test


def mean_std_matrix(all_res_mean, mean_stds):
    std_array = np.empty(dtype=object, shape=all_res_mean.shape)
    for i in range(all_res_mean.shape[0]):
        for j in range(all_res_mean.shape[1]):
            for k in range(all_res_mean.shape[2]):
                std_array[i, j, k] = (f"{all_res_mean[i, j, k]:.3f}" + u"\u00B1" + f"{mean_stds[i, j, k]:.3f}")
    return std_array


def model_predict(exp_type, param, train_array, y_train, test_array,random_state):
    if exp_type == "python":
        clf = DecisionTreeRegressor(criterion=param['criterion'],
                                    max_depth=param['max_depth'],
                                    min_samples_split=param['min_samples_split'],
                                    min_samples_leaf=param['min_samples_leaf'],
                                    max_features=param['max_features'],
                                    random_state=random_state)
        clf.fit(train_array, y_train)
        return clf.predict(test_array)
    else:
        estimator = MaxMarginIntervalTree(margin=param['margin'],  max_depth=param['max_depth'],  min_samples_split=param['min_samples_split'], loss = 'linear_hinge')    
        out =estimator.fit(train_array.values, y_train)
        out_pruning= min_cost_complexity_pruning(estimator)
        preds = estimator.predict(test_array.values)
        return preds


def cross_val(n_splits, random_state, X_array, exp, y_interval, parameters_tree, mean_array=None, exp_type="python"):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(X_array)
    param_list = list(ParameterGrid(parameters_tree))
    r2_matrix = np.zeros((n_splits, len(param_list)))
    mse_matrix = np.zeros((n_splits, len(param_list)))
    int_matrix = np.zeros((n_splits, len(param_list)))
    r2_int_matrix = np.zeros((n_splits, len(param_list)))
    z0_matrix = np.zeros((n_splits, len(param_list)))
    for i, (train_index, test_index) in enumerate(kf.split(X_array)):
        train_array, test_array, y_train, y_test, y_int_test = test_train_split(X_array, exp, y_interval, train_index,
                                                                                test_index, mean_array)
        for j in range(len(param_list)):
            y_pred = model_predict(exp_type, param_list[j], train_array, y_train, test_array,random_state)

            mse = ((y_test - y_pred) ** 2).sum()
            mse_matrix[i, j] = np.sqrt(mse / y_test.shape[0])
            total_sum_of_squares = ((y_test - y_test.mean()) ** 2).sum()
            r2_matrix[i, j] = 1 - (mse / total_sum_of_squares)
            int_mse = interval_mse(y_int_test, y_pred)
            int_matrix[i, j] = np.sqrt(int_mse / y_test.shape[0])
            int_tss = interval_mse(y_int_test, [y_test.mean()] * y_test.shape[0])
            r2_int_matrix[i, j] = 1 - (int_mse / int_tss)
            z0_matrix[i, j] = zero_one_loss(y_int_test, y_pred)

    argmax_r2 = np.argmax(np.mean(r2_matrix, axis=0))
    argmax_int_r2 = np.argmax(np.mean(r2_int_matrix, axis=0))
    argmax_z0 = np.argmax(np.mean(z0_matrix, axis=0))

    max_r2 = np.mean(r2_matrix, axis=0)[argmax_r2]
    std_r2 = np.std(r2_matrix, axis=0)[argmax_r2]
    min_mse = np.mean(mse_matrix, axis=0)[argmax_r2]
    std_mse = np.std(mse_matrix, axis=0)[argmax_r2]

    max_r2_int = np.mean(r2_int_matrix, axis=0)[argmax_int_r2]
    std_int_r2 = np.std(r2_int_matrix, axis=0)[argmax_int_r2]
    min_mse_int = np.mean(int_matrix, axis=0)[argmax_int_r2]
    std_int_mse = np.std(int_matrix, axis=0)[argmax_int_r2]
    max_z0 = np.mean(z0_matrix, axis=0)[argmax_z0]
    std_z0 = np.std(z0_matrix, axis=0)[argmax_z0]

    if exp_type == "python":
        clf_best_r2 = DecisionTreeRegressor(criterion=param_list[argmax_r2]['criterion'],
                                            max_depth=param_list[argmax_r2]['max_depth'],
                                            min_samples_split=param_list[argmax_r2]['min_samples_split'],
                                            min_samples_leaf=param_list[argmax_r2]['min_samples_leaf'],
                                            max_features=param_list[argmax_r2]['max_features'],
                                            random_state=random_state)
        clf_best_int = DecisionTreeRegressor(criterion=param_list[argmax_int_r2]['criterion'],
                                             max_depth=param_list[argmax_int_r2]['max_depth'],
                                             min_samples_split=param_list[argmax_int_r2]['min_samples_split'],
                                             min_samples_leaf=param_list[argmax_int_r2]['min_samples_leaf'],
                                             max_features=param_list[argmax_int_r2]['max_features'],
                                             random_state=random_state)
    else:
        # code MMIT ici
        clf_best_r2 = MaxMarginIntervalTree(margin=param_list[argmax_r2]['margin'],
                                                    max_depth=param_list[argmax_r2]['max_depth'], 
                                                    min_samples_split=param_list[argmax_r2]['min_samples_split'], 
                                                    loss = 'linear_hinge')
        clf_best_int = MaxMarginIntervalTree(margin=param_list[argmax_int_r2]['margin'],
                                                    max_depth=param_list[argmax_int_r2]['max_depth'], 
                                                    min_samples_split=param_list[argmax_int_r2]['min_samples_split'], 
                                                    loss = 'linear_hinge')

    if len(exp.shape) == 1:
        clf_best_r2.fit(X_array, exp)
        r2_score = clf_best_r2.score(X_array, mean_array)
        number_of_leaf = len(np.where(clf_best_r2.tree_.children_left == -1)[0])
        clf_best_int.fit(X_array, exp)
        r2_int_score = clf_best_int.score(X_array, mean_array)
    elif exp.shape[1] == 3:
        train_array = pd.concat([X_array] * 3, ignore_index=True, axis=0)
        y_array = pd.concat([exp.iloc[:, 0], exp.iloc[:, 1], exp.iloc[:, 2]], ignore_index=True, axis=0)
        clf_best_r2.fit(train_array, y_array)
        r2_score = clf_best_r2.score(X_array, mean_array)
        number_of_leaf = len(np.where(clf_best_r2.tree_.children_left == -1)[0])
        clf_best_int.fit(train_array, y_array)
        r2_int_score = clf_best_int.score(X_array, mean_array)
    else:
        # arbres MMIT
        clf_best_r2.fit(X_array, exp)
        clf_best_int.fit(X_array, exp)
        number_of_leaf = len(clf_best_r2.tree_.leaves)
        r2_score = metrics.r2_score(clf_best_r2.predict(X_array), mean_array)
        r2_int_score = metrics.r2_score(clf_best_int.predict(X_array), mean_array)
    return (max_r2, min_mse, max_r2_int, min_mse_int, max_z0), (
            std_r2, std_mse, std_int_r2, std_int_mse, std_z0), clf_best_r2, clf_best_int, number_of_leaf, (
           r2_score, r2_int_score)

def experiments(array,test_cols, exp_vals):
    all_res = np.zeros((5, 5, 8))
    standard_devs = np.zeros((5, 5, 8))
    leaf_nums = np.zeros((8, 5))
    best_r2_clf = []
    best_int_clf = []
    train_scores = np.zeros((8, 2, 5))
    for depth in [1, 2, 3, 4, 5, 6, 7, 8]:
        for experience in [1, 2, 3, 4, 5]:
            print(experience)
            X_array = pd.DataFrame(array[test_cols])
            if exp_vals != 'mmit':
                parameters_tree = {'criterion': ["squared_error"],
                                    'margin' : [0, 1, 2],
                                'max_depth': [depth],
                                'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                'max_features': [None]}
            else:
                parameters_tree = {'criterion': ["squared_error"],
                                'margin' : [0, 1, 2],
                                'max_depth': [depth],
                                'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                'max_features': [None]}
            random_state = check_random_state(42)

            if experience == 1:
                exp_array = [array[f"Rep{i}"] for i in range(1, 4)]
            elif experience == 2:
                exp_array = [array[f"Rep{i}.1"] for i in range(1, 4)]
            elif experience == 3:
                exp_array = [array[f"Rep{i}.2"] for i in range(1, 4)]
            elif experience == 4:
                exp_array = [array[f"Rep{i}.3"] for i in range(1, 4)]
            elif experience == 5:
                exp_array = [array[f"Rep{i}.4"] for i in range(1, 4)]

            exp, mean_exp, y_interval = y_array(exp_array, exp_vals)
            exp_type = "python" if exp_vals != "mmit" else "mmit"
            results, stds, best_param_r2, best_int_param, numleaf, train_score = cross_val(n_splits=5,
                                                                                 random_state=random_state,
                                                                                 X_array=X_array,
                                                                                 exp=exp,
                                                                                 y_interval=y_interval,
                                                                                 parameters_tree=parameters_tree,
                                                                                 mean_array=mean_exp,
                                                                                exp_type=exp_type)
            best_r2_clf.append(best_param_r2)
            best_int_clf.append(best_int_param)
            all_res[experience - 1, :, depth - 1] = results
            standard_devs[experience - 1, :, depth - 1] = stds
            leaf_nums[depth - 1, experience - 1] = numleaf
            train_scores[depth - 1, :, experience - 1] = train_score
    return all_res, standard_devs, leaf_nums, best_r2_clf, best_int_clf, train_scores

def separate_peptides(array, tree):
    liste = []
    for i in range(array.shape[0]):
        X = array.iloc[i, :]
        index = 0
        while index != -1:
            feature = tree.feature[index]
            threshold = tree.threshold[index]
            if X.iloc[feature] <= threshold:
                child_index = tree.children_left[index]
            else:
                child_index = tree.children_right[index]
            if child_index == -1:
                liste.append((i, index))
            index = child_index

    dictionary = {}
    for p in liste:
        if not p[1] in dictionary.keys():
            dictionary[p[1]] = [p[0]]
        else:
            dictionary[p[1]].append(p[0])
    return dictionary