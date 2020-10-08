import pandas as pd
import math
import csv

path = "C:/workspaces/python-project/yong-in/data/"

year = 2018

csv_path = path + str(year) + ".csv"
# 한글이 있어 오류가 생기므로 engine='python' 을 코드에 넣어줌
df = pd.read_csv(csv_path, engine='python')

df_np = df.values

start = 1416
end = 6553



# 3월1일 데이터에서 월, 일 정보 추출
tmp = str(df_np[start][1])
month = int(tmp[6:7])
day = int(tmp[8:10])

sum_of_temp = 0  # 온도 합
sum_of_rain = 0  # 강우량 합
sum_of_hum = 0  # 습도 합

# 하루 평균으로 할 것이므로 index 를 따로 만듬(Nan 이 있을 때, 평균을 잘못 구함을 방지)
idx_temp = 0
idx_hum = 0


with open("modifided_2018.csv", 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    wr.writerow(['month', 'day', "average of temperature", 'sum of rainfall', 'average of humidity'])

    for i in range(start, end):
        tmp_str = str(df_np[i][1])
        if month == int(tmp_str[6:7]) and day == int(tmp_str[8:10]):  # 월과 일이 동일 하다면

            if math.isnan(df_np[i][2]):
                pass
            else:
                sum_of_temp += df_np[i][2]
                idx_temp += 1

            if math.isnan(df_np[i][3]):
                pass
            else:
                sum_of_rain += df_np[i][3]
                # 강수량은 합으로 하기 때문에 평균을 이용할 필요 없음.

            if math.isnan(df_np[i][4]):
                pass
            else:
                sum_of_hum += df_np[i][4]
                idx_hum += 1
        else:
            wr.writerow([month, day, round(sum_of_temp / idx_temp, 2), round(sum_of_rain, 2), round(sum_of_hum / idx_hum, 2)])

            print(month, "월", day, "일")
            print("average of temperature", round(sum_of_temp / idx_temp, 2))
            print("rainfall : ", round(sum_of_rain, 2))  # float 의 경우 소수점 밑으로 값이 쭉 내려가는 것 방지
            print("average of humidity :", round(sum_of_hum / idx_hum, 2))
            print("============================================================")

            # 다음 날의 월, 일 정보 저장
            month = int(tmp_str[6:7])
            day = int(tmp_str[8:10])

            # 인덱스, 합 값 초기화
            sum_of_temp = 0  # 온도 합
            sum_of_rain = 0  # 강우량 합
            sum_of_hum = 0  # 습도 합

            idx_temp = 1
            idx_hum = 1
            print("날짜가 바뀜으로 인한 인덱스 초기화 완료")

            if math.isnan(df_np[i][2]):
                pass
            else:
                sum_of_temp += df_np[i][2]

            if math.isnan(df_np[i][3]):
                pass
            else:
                sum_of_rain += df_np[i][3]

            if math.isnan(df_np[i][4]):
                pass
            else:
                sum_of_hum += df_np[i][4]

