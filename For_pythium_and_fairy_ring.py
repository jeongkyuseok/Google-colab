from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier


from random import *

# from xgboost import plot_importance
#
# from matplotlib import pyplot as plt
#
# from sklearn.externals import joblib

import pandas as pd
import numpy as np
import time

data_path = "C:/workspaces/python-project/yong-in/solve_imbalanced/over_sampled_data_set/"

diseases = ["fairy_ring", "pythium_blight", "pythium_root_dysfunction"]


To_path = "C:/workspaces/python-project/yong-in/modeling/trained_models/"

max_array = []

for dis in diseases:
    print("Using Random Trees Embedding")
    print("============================================")
    print(dis)

    csv_path = data_path + "over_sampled_" + dis + ".csv"

    dis_csv = pd.read_csv(csv_path)

    csv_np = dis_csv.values

    X = csv_np[:,:-1]
    y_2d = csv_np[:,-1:]

    # Theses 3 lines code make 2d-array(say, y) 1d-array
    y = np.array([])
    for num in y_2d:
        y = np.append(y, num)

    max = 0
    max_trees = 0
    max_lr = 0

    # Split data to train-set and test-data, ratio : 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for j in range(100):
        # print("********************************")
        # num_of_tree = randint(200, 500)
        # print("Number of Tree : {}".format(num_of_tree))

        # forest = RandomForestClassifier(criterion='gini',
        #                                 max_features="auto",
        #                                 n_estimators=num_of_tree,
        #                                 random_state=15)

        # forest.fit(X_train, y_train)
        #
        # print("Random Forest Score : {}".format(forest.score(X_test, y_test)))

        # for i in range(0, 4):
        #
        #     lr = uniform(1e-5, 1)
        #
        #     # gbm = GradientBoostingClassifier(n_estimators=num_of_tree,
        #     #                              learning_rate = lr)
        #     # gbm.fit(X_train, y_train)
        #     # print("{}, Score : {}, learning_rate : {}".format(i, gbm.score(X_test, y_test), lr))
        #
        #     xgb = XGBClassifier(learning_rate=lr)
        #     xgb.fit(X_train, y_train)
        #
        #     print("{}, Score : {}, learning_rate : {}".format(i, xgb.score(X_test, y_test), lr))
        #     print("-----------------------")

        # lr = uniform(0.7, 1.08)
        #
        # xgb = XGBClassifier(learning_rate=lr)
        # xgb.fit(X_train, y_train)
        # score = xgb.score(X_test, y_test)
        #
        # if max <= score:
        #     print("{}, Score : {}, learning_rate : {}    MAX! ".format(j, score, lr))
        #     max = score
        # else:
        #     print("{}, Score : {}, learning_rate : {}".format(j, score, lr))
        #
        # print("-----------------------")

        start = time.time()

        lr = uniform(1e-4, 1)

        num_of_tree = randint(10, 500)

        bc = BaggingClassifier(n_estimators=num_of_tree,  random_state=777)

        bc.fit(X_train, y_train)

        score = bc.score(X_test, y_test)

        if max <= score:
            print("{}, learning_rate : {}, num_of_trees : {}, score : {}    MAX! ".format(j, lr, num_of_tree, score))
            max = score
            max_trees = num_of_tree
            max_lr = lr
        else:
            print("{}, learning_rate : {}, num_of_trees : {}, score : {}".format(j, lr, num_of_tree, score))
        print("수행시간 : {0:.2f} 초".format( time.time()-start ) )


        # num_of_tree = randint(10, 500)
        # etc = ExtraTreesClassifier(n_estimators=num_of_tree, bootstrap=True, n_jobs=4)
        #
        # etc.fit(X_train, y_train)
        # score = etc.score(X_test, y_test)
        #
        # if max <= score:
        #     print("{}, Score : {}, num_of_trees : {}    MAX! ".format(j, score, num_of_tree))
        #     max = score
        #     max_trees = num_of_tree
        # else:
        #     print("{}, Score : {}, num_of_trees : {}".format(j, score, num_of_tree))
        # print("-----------------------")

        # num_of_tree = randint(10, 11)
        # isof = IsolationForest(n_estimators=num_of_tree, bootstrap=True, n_jobs=4, random_state=777)
        #
        # isof.fit(X_train, y_train)
        # score = isof.score_samples(X_test)
        #
        # print(score)

        # if max <= score:
        #     print("{}, Score : {}, num_of_trees : {}    MAX! ".format(j, score, num_of_tree))
        #     max = score
        #     max_trees = num_of_tree
        # else:
        #     print("{}, Score : {}, num_of_trees : {}".format(j, score, num_of_tree))
        # print("-----------------------")

    max_array.append([dis, max, max_trees, max_lr])

# for arr in max_array:
#     print(arr[0] + " - number of trees : {}, score : {}".format(arr[2], arr[1]))

for arr  in max_array:
    print(arr[0] + " - number of trees : {}, learning_rate : {}, score : {}".format(arr[2], arr[3], arr[1]))


    # joblib.dump(forest, To_path + dis + '_model.pkl')



    # plot_importance(xgb)
    # plt.show()

