import pandas as pd
import math

import csv

# print 했을 때 다보이는 방법
pd.set_option("display.max_rows", 10000)

path = "C:/workspaces/python-project/yong-in/data/"

year = 2011
start = 4344
end = 6535


csv_path = path+str(year)+".csv"
# 한글이 있어 오류가 생기므로 engine='python' 을 코드에 넣어줌
df = pd.read_csv(csv_path,header=1,engine='python')

df_np = df.values

for data in df_np:
    print(data)