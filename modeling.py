from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.externals import joblib

import pandas as pd
import numpy as np




data_path = "C:/workspaces/python-project/yong-in/solve_imbalanced/over_sampled_data_set/"

diseases=["anthracnose", "brown_patch", "dollar_spot", "fairy_ring", "large_patch",
         "leaf_spot", "pythium_blight", "pythium_root_dysfunction", "summer_patch", "yellow_patch"]

S_diseases = ["fairy_ring", "pythium_blight", "pythium_root_dysfunction"]

To_path = "C:/workspaces/python-project/yong-in/modeling/trained_models/"

s_dis_tree_num = {"fairy_ring" : 113, "pythium_blight" : 459, "pythium_root_dysfunction" : 417}

for dis in diseases:
    num_of_tree = 100

    csv_path = data_path + "over_sampled_" + dis + ".csv"

    dis_csv = pd.read_csv(csv_path)

    csv_np = dis_csv.values

    X = csv_np[:, :-1]
    y_2d = csv_np[:, -1:]

    # Theses 3 lines code make 2d-array(say, y) 1d-array
    y = np.array([])
    for num in y_2d:
        y = np.append(y, num)

    # Split data to train-set and test-data, ratio : 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if dis in S_diseases:  ## 페어리링, 피시움성 병해
        num_of_tree = int( s_dis_tree_num.get(dis) )
        print(num_of_tree)

        bc = BaggingClassifier(n_estimators = num_of_tree, random_state=777)
        bc.fit(X_train, y_train)     ## 훈련

        print(dis + " score : {}".format( bc.score(X_test, y_test)  ))    ## 훈련한 모델의 정확도 테스트

        joblib.dump(bc, To_path + dis + '_model.pkl')  ## 훈련한 모델 저장


    else:
        forest = RandomForestClassifier(n_estimators = num_of_tree, random_state = 777)
        forest.fit(X_train, y_train)

        print(dis + " score : {}".format(forest.score(X_test, y_test)) )

        joblib.dump(forest, To_path + dis + '_model.pkl')

    # print(dis," 의 학습 소요 시간 : ", time.time() - start)

"""
    저장된 모델을 load 해 예측
"""
for i, dis in enumerate(diseases):

    file_name = To_path + dis + '_model.pkl'

    loaded_model = joblib.load(file_name)

    # 입력 값은 무조건 2D-array
    X_exam = np.array([[0,0,0,0]])

    print( "[[0, 0, 0, 0]] predict : ", loaded_model.predict(X_exam))
