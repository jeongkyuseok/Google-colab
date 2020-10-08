import os
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.externals import joblib

file_name = "outfile_2019112012.csv"
csv_path = "C:/Users/user/Desktop/Calmet/" + file_name


"""
    날씨 데이터(csv파일) 가져오기
    
    date 변수는 가지고 올 날짜 데이터의 폴더로 
    
    YYYYmmDD00 또는 YYYYmmDD12 같은 방식으로 표기됩니다. 
    
    여기서 앞의 YYYY는 년도 mm은 월 DD은 일 이며
    
    맨 뒤자리 00은 YYYYmmDD 일자 오후 8시 34분경에 데이터가 생성되며 
    
    데이터의 시간 범위는 DD일 오전 9시 ~ DD+2일 오후 7시까지
    
    12은 YYYYmmDD 일자 오전 8시 34분경에 데이터가 생성되며 
    
    데이터의 시간 범위는 DD일 오후 9시 ~ DD+2일 오전 7시까지입니다.     
"""


def get_weather_data():
    """
        csv파일의 첫 전처리 단계로  csv파일을 받아서
        날씨 데이터를 numpy 배열로 바꿔주는 함수입니다.
    """

    # path_dir = "C:/Users/cjdru/OneDrive/바탕 화면/new\weather/"  # 날씨 데이터가 있는 폴더로 수정이 필요한 부분입니다.
    #
    # file_list = os.listdir(path_dir)  # 해당 폴더에 있는 파일 리스트
    #
    # file = file_list[0]  # 각 폴더에 하나의 csv파일만 있습니다.
    #
    # csv_path = path_dir + "/" + file

    df = pd.read_csv(csv_path)

    df_np = df.values

    return df_np


def seperate_per_3time(df_np):
    """
        3시간 단위( 21~23시 0시~2시 등 )으로 기온, 토양온도, 강수량, 습도 출력 15개의 2차원 numpy array가 리스트로 출력 될 것이다.(각 원소가 2차원 numpy array)
        또한 해솔리아의 코스만 적용했습니다.

        만약 2019111300이면 반환되는 data_list의 0번째 인덱스의 원소는 21시~23시, 1번 인덱서 원소는 0~2시에 관한 데이터입니다.
    """
    data_list = []

    tmp_arr = np.full((54, 4), 0, dtype=float)
    discriminator = 0

    for i, data in enumerate(df_np):
        q, r = divmod(i, 54)
        if q % 2 != 0:
            pass
        else:  # 해솔리아만 해당
            time = data[2]
            tmp = time // 3  # 몫
            if discriminator != tmp:
                data_list.append(tmp_arr)
                tmp_arr = np.full((54, 4), 0, dtype=float)
                discriminator = tmp

            # 기온 0  토양 1  강수량 2   습도 3 in tmp_arr
            # data 7-기온  9-토양온도   8-강수량   10 - 습도
            if discriminator < 15:
                tmp_arr[r][0] = tmp_arr[r][0] + round(data[7] / 3, 3)
                tmp_arr[r][1] = tmp_arr[r][1] + round(data[9] / 3, 3)
                tmp_arr[r][2] = tmp_arr[r][2] + data[8]
                tmp_arr[r][3] = tmp_arr[r][3] + data[10] / 3
            else:
                tmp_arr[r][0] = tmp_arr[r][0] + round(data[7] / 2, 3)
                tmp_arr[r][1] = tmp_arr[r][1] + round(data[9] / 2, 3)
                tmp_arr[r][2] = tmp_arr[r][2] + data[8]
                tmp_arr[r][3] = tmp_arr[r][3] + data[10] / 2

        if i == 5075:
            data_list.append(tmp_arr)

    return data_list


def get_incidence(date):
    """
        date는 numpy배열로 2차원이여야 합니다. ex) X = np.array([[13,16,81,16]])

        해당 병해의 순서는 이전에 알려드렸던 병해 표출 순서와 동일합니다.
        또한 최종으로 시간대별 지역별로 반환되는 발병률 리스트의 값순서도 동일합니다.
    """
    diseases = ["fairy_ring", "large_patch", "pythium_blight", "pythium_root_dysfunction",
                "anthracnose", "brown_patch", "dollar_spot", "leaf_spot", "summer_patch", "yellow_patch"]

    # 발병률을 리스트로 저장(리스트로 반환 예정)
    incidence = []
    for dis in diseases:
        To_path = "C:/workspaces/python-project/yong-in/modeling/trained_models/"  # pkl 파일이 있는 폴더로 수정이 필요한 부분입니다. (마지막에 / 로 끝나게 해주세요.)
        file_name = To_path + dis + "_model.pkl"
        loaded_model = joblib.load(file_name)
        incidence.append(loaded_model.predict(date)[0])

    return incidence


def show_incidence(pre_data):
    final_result = []
    """
        해당 함수는 최종으로 결과를 표출 할 함수입니다. 
        
        result는 리스트로 총 16개의 원소를 가지며 첫번째 원소는 21~23시(9~11시), 두번째는 0~2시(12~14시) ... 
        마지막 원소는 18~19시(6~7시) 병해 발병률이 들어있습니다.
        각 원소도 리스트이며 첫번째 원소는 hf11에서 각 병해 별 발병률 리스트(ex [0, 0, 0, 10, 0, 0, 0, 0, 0, 10]로 표현) 부터  
        hg39까지 발병률 리스트가 있습니다. 
    """
    for i, data in enumerate(pre_data):
        tmp_list = []
        for low_data in data:
            tmp_list.append(get_incidence(np.array([low_data])))

        print(i)
        # if i >= 2:
        #     break
        final_result.append(tmp_list)

    return final_result


def data_format(result,name):
    fairy = []
    large = []
    pythium = []
    pythium_root = []
    tan = []
    brown = []
    spot = []
    yepgo = []
    summer = []
    yellow = []
    total_info = []
    total = []
    for data in result:
        fairy.append(data[0])
        large.append(data[1])
        pythium.append(data[2])
        pythium_root.append(data[3])
        tan.append(data[4])
        brown.append(data[5])
        spot.append(data[6])
        yepgo.append(data[7])
        summer.append(data[8])
        yellow.append(data[9])

    total_info.append({
        'fairy' : fairy,
        'large' : large,
        'pythium' : pythium,
        'pythium_root' : pythium_root,
        'tan' : tan,
        'brown' : brown,
        'spot' : spot,
        'yepgo' : yepgo,
        'summer' : summer,
        'yellow' : yellow


    })
    total.append({
        name : total_info,

    })
    return total


def main():
    print("start")
    weather_data = get_weather_data()

    pre_data = seperate_per_3time(weather_data)

    rate = show_incidence(pre_data)
    hae1 = []
    hae2 = []
    hae3 = []
    hae4 = []
    hae5 = []
    hae6 = []
    hae7 = []
    hae8 = []
    hae9 = []

    sol1 = []
    sol2 = []
    sol3 = []
    sol4 = []
    sol5 = []
    sol6 = []
    sol7 = []
    sol8 = []
    sol9 = []

    ria1 = []
    ria2 = []
    ria3 = []
    ria4 = []
    ria5 = []
    ria6 = []
    ria7 = []
    ria8 = []
    ria9 = []

    fairy_max = [-1.0, -1.0, -1.0]
    large_max = [-1.0, -1.0, -1.0]

    for idx1,result in enumerate(rate):
        for idx2, data_1 in enumerate(result):
            if idx2 == 9 :
                hae1.append(data_1)
            elif idx2 == 10 :
                hae2.append(data_1)
            elif idx2 == 11 :
                hae3.append(data_1)
            elif idx2 == 12 :
                hae4.append(data_1)
            elif idx2 == 13 :
                hae5.append(data_1)
            elif idx2 == 14 :
                hae6.append(data_1)
            elif idx2 == 15 :
                hae7.append(data_1)
            elif idx2 == 16 :
                hae8.append(data_1)
            elif idx2 == 17 :
                hae9.append(data_1)
            elif idx2 == 27 :
                sol1.append(data_1)
            elif idx2 == 28:
                sol2.append(data_1)
            elif idx2 == 29 :
                sol3.append(data_1)
            elif idx2 == 30 :
                sol4.append(data_1)
            elif idx2 == 31 :
                sol5.append(data_1)
            elif idx2 == 32 :
                sol6.append(data_1)
            elif idx2 == 33 :
                sol7.append(data_1)
            elif idx2 == 34 :
                sol8.append(data_1)
            elif idx2 == 35 :
                sol9.append(data_1)
            elif idx2 == 45 :
                ria1.append(data_1)
            elif idx2 == 46 :
                ria2.append(data_1)
            elif idx2 == 47 :
                ria3.append(data_1)
            elif idx2 == 48 :
                ria4.append(data_1)
            elif idx2 == 49 :
                ria5.append(data_1)
            elif idx2 == 50 :
                ria6.append(data_1)
            elif idx2 == 51 :
                ria7.append(data_1)
            elif idx2 == 52 :
                ria8.append(data_1)
            elif idx2 == 53 :
                ria9.append(data_1)

            # if idx1 < 2:
            #     if fairy_max[0] < data_1[0]:
            #         fairy_max[0] = data_1[0]
            #
            #     if large_max[0] < data_1[1]:
            #         large_max[0] = data_1[1]
            # elif idx1 >=2 and idx1 < 9:
            #     if fairy_max[1] < data_1[0]:
            #         fairy_max[1] = data_1[0]
            #
            #     if large_max[1] < data_1[1]:
            #         large_max[1] = data_1[1]
            # else:
            #     if fairy_max[2] < data_1[0]:
            #         fairy_max[2] = data_1[0]
            #
            #     if large_max[2] < data_1[1]:
            #         large_max[2] = data_1[1]

            if idx1 < 6:
                if fairy_max[0] < data_1[0]:
                    fairy_max[0] = data_1[0]

                if large_max[0] < data_1[1]:
                    large_max[0] = data_1[1]
            elif idx1 >=6 and idx1 < 14:
                if fairy_max[1] < data_1[0]:
                    fairy_max[1] = data_1[0]

                if large_max[1] < data_1[1]:
                    large_max[1] = data_1[1]
            else:
                if fairy_max[2] < data_1[0]:
                    fairy_max[2] = data_1[0]

                if large_max[2] < data_1[1]:
                    large_max[2] = data_1[1]

        print(result)
        #09,12,15,18,21,00,03,06,09,12,15,18,21,00,03,06

    for i in hae1:
        print("hae1 : " + str(i))

    for i in fairy_max :
        print("fairy : " + str(i))

    for i in large_max :
        print("large : " + str(i))

    machine_info = []
    hae1 = data_format(hae1, 'hae1')
    hae2 = data_format(hae2, 'hae2')
    hae3 = data_format(hae3, 'hae3')
    hae4 = data_format(hae4, 'hae4')
    hae5 = data_format(hae5, 'hae5')
    hae6 = data_format(hae6, 'hae6')
    hae7 = data_format(hae7, 'hae7')
    hae8 = data_format(hae8, 'hae8')
    hae9 = data_format(hae9, 'hae9')

    machine_info.append(
        {
            'hae': hae1 + hae2+ hae3+ hae4+ hae5+ hae6+ hae7+ hae8+ hae9,
            'sol': data_format(sol1,'sol1') + data_format(sol2,'sol2')+ data_format(sol3,'sol3')+ data_format(sol4,'sol4')+ data_format(sol5,'sol5')+ data_format(sol6,'sol6')+ data_format(sol7,'sol7')+ data_format(sol8,'sol8')+ data_format(sol9,'sol9'),
            'ria': data_format(ria1,'ria1') + data_format(ria2,'ria2')+ data_format(ria3,'ria3')+ data_format(ria4,'ria4')+ data_format(ria5,'ria5')+ data_format(ria6,'ria6')+ data_format(ria7,'ria7')+ data_format(ria8,'ria8')+ data_format(ria9,'ria9'),
            'fairy_max' : fairy_max,
            'large_max' : large_max
        }

    )

    print(machine_info)

    with open('E:/output_file/out.txt', 'w') as f:
        json.dump(machine_info, f)


    return machine_info



if __name__ == '__main__':
    main()
# 'sol': data_format(sol1) + data_format(sol2)+ data_format(sol3)+ data_format(sol4)+ data_format(sol5)+ data_format(sol6)+ data_format(sol7)+ data_format(sol8)+ data_format(sol9),
# 'ria': data_format(ria1) + data_format(ria2)+ data_format(ria3)+ data_format(ria4)+ data_format(ria5)+ data_format(ria6)+ data_format(ria7)+ data_format(ria8)+ data_format(ria9),