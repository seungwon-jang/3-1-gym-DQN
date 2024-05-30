#csv파일 읽어와서 거기에 데이터 추가하는 것을 연습하는 파일

import pandas as pd

epi_data = {'epi' : [], 'end_time_step' : [], 'total_return' : []}
data_df = pd.DataFrame(epi_data)            #학습 데이터를 저장할 데이터 프레임

for i in range(100):
    epi, Return = i, i
    epi_df = pd.DataFrame({'epi' : [epi], 'end_time_step' : [i], 'total_return' : [Return]})
    data_df = pd.concat([data_df, epi_df], ignore_index= True)
print(data_df)

data_df.to_csv('my_data.csv', mode='a', header=False, index=False)