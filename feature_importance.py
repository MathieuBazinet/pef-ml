import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor
import mmit

array = pd.read_csv("./csv_laurent.csv")

importance_cols = ['Mol_weight', 'Isoelectric_point', 'GRAVY Score', 'Y', 'H', 'L', 'P', 'R', 'E']
X_importance = pd.DataFrame(array[importance_cols])
panda_list = []
for experience in range(1,6):
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

  exp = np.mean(pd.concat(exp_array, ignore_index=True, axis=1),axis=1)
  random_state = check_random_state(42)
  clf0 = RandomForestRegressor(n_estimators=1000, random_state=random_state)
  clf0 = clf0.fit(np.array(X_importance.values), exp)
  rank = np.argsort(clf0.feature_importances_)[::-1]
  panda_list.append(pd.DataFrame(np.vstack((np.array(importance_cols,dtype=object)[rank],
                                            clf0.feature_importances_[rank]))))
  print("Expérience", experience, "Somme des trois premières importances",
        np.sum(clf0.feature_importances_[rank][0:3]))

print(pd.concat(panda_list, axis=0))