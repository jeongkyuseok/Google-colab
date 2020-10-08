from sklearn.externals import joblib
import numpy as np


diseases=["anthracnose", "brown_patch", "dollar_spot", "fairy_ring", "large_patch",
         "leaf_spot", "pythium_blight", "pythium_root_dysfunction", "summer_patch", "yellow_patch"]

To_path = "C:/workspaces/python-project/yong-in/modeling/trained_models/"

print( "기온 : 13, 토양온도 : 90, 강수량 : 80, 습도 : 16 ")
for dis in diseases:

    file_name = To_path + dis + '_model.pkl'

    loaded_model = joblib.load(file_name)

    # 입력 값은 무조건 2D-array
    X_exam = np.array([[9.21,13.21,0,57.388]])

    print( dis + " predict : ", loaded_model.predict(X_exam)[0])