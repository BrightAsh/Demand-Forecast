import os
import pandas as pd
import matplotlib.pyplot as plt

korean_holidays = pd.to_datetime([
    '2021-01-01',  # 신정
    '2021-02-11', '2021-02-12', '2021-02-13',  # 설날 연휴
    '2021-03-01',  # 삼일절
    '2021-05-05',  # 어린이날
    '2021-05-19',  # 석가탄신일
    '2021-06-06',  # 현충일
    '2021-08-15',  # 광복절
    '2021-09-20', '2021-09-21', '2021-09-22',  # 추석 연휴
    '2021-10-03',  # 개천절
    '2021-10-09',  # 한글날
    '2021-12-25',  # 성탄절
    '2022-01-01',  # 신정
    '2022-01-31', '2022-02-01', '2022-02-02',  # 설날 연휴
    '2022-03-01',  # 삼일절
    '2022-03-09',   # 제20대 대통령 선거일
    '2022-05-05',  # 어린이날
    '2022-05-08',  # 석가탄신일
    '2022-06-01', # 제8회 전국동시지방선거일
    '2022-06-06',  # 현충일
    '2022-08-15',  # 광복절
    '2022-09-09', '2022-09-10', '2022-09-11','2022-09-12',  # 추석 연휴& 대체공휴일
    '2022-10-03',  # 개천절
    '2022-10-09',  # 한글날
    '2022-10-10',  # 한글날 대체 공휴일
    '2022-12-25',  # 성탄절
    '2023-01-01',  # 신정
    '2023-01-21', '2023-01-22', '2023-01-23','2023-01-24',  # 설날 연휴
    '2023-03-01',  # 삼일절
    '2023-05-05',  # 어린이날
    '2023-05-27',  # 석가탄신일
    '2023-06-06',  # 현충일
    '2023-08-15',  # 광복절
    '2023-09-28', '2023-09-29', '2023-09-30',  # 추석 연휴
    '2023-10-03',  # 개천절
    '2023-10-09',  # 한글날
    '2023-12-25',  # 성탄절
    '2024-01-01',  # 신정
    '2024-02-09', '2024-02-10', '2024-02-11',  # 설날 연휴
    '2024-02-12',  # 설날 대체 공휴일 (일요일에 겹침)
    '2024-03-01',  # 삼일절
    '2024-04-10',  # 총선
    '2024-05-05',  # 어린이날
    '2024-05-06',  # 어린이날 대체 공휴일 (일요일에 겹침)
    '2024-05-15',  # 석가탄신일
    '2024-06-06'  # 현충일
])



pd.set_option('display.max_columns', None)
# 데이터 load
path = 'C:/Users/User/Desktop/Project/유통/수요예측/중소유통물류센터 거래 데이터/'
all_files = os.listdir(path)
xlsx_files = [f for f in all_files if f.endswith('.xlsx')]
root_path ='C:/Users/User/Desktop/Project/유통/수요예측/우편번호 크롤링/'


##############################################################################################################################
# data2 전처리 및 데이터 분석 (면류,라면류)
all_sheets = pd.read_excel(os.path.join(path, xlsx_files[1]), sheet_name=None)
combined_data = pd.concat(all_sheets.values(), ignore_index=True)
save_path='C:/Users/User/Desktop/Project/유통/수요예측/2 분석결과/'

##########################################################################################################################################################
#종류 별로 그래프
data1=combined_data.copy()
data1['연도'] = data1['판매일'].dt.year
data1['월'] = data1['판매일'].dt.month
data1['연월'] = data1['판매일'].dt.to_period('M')
grouped_data = data1.groupby(['대분류', '연월'])['판매수량'].sum().reset_index()
unique_categories = grouped_data['대분류'].unique()
plt.figure(figsize=(10, 6))
for category in unique_categories:
    if category not in ['기타']:
        category_data = grouped_data[grouped_data['대분류'] == category]
        plt.plot(category_data['연월'].astype(str), category_data['판매수량'], marker='o', label=category)
plt.title('대분류별 판매수량 추이')
plt.xlabel('연월')
plt.ylabel('판매수량 합계')
plt.xticks(rotation=45)
plt.legend(title='대분류')
plt.grid(True)
plt.savefig(save_path + 'kind.png')
plt.close()

##########################################################################################################################################################
#상관계수 구하기
data2=combined_data.copy()
data2['연도'] = data2['판매일'].dt.year
data2['월'] = data2['판매일'].dt.month
grouped_data = data2.groupby(['대분류', '연도', '월'])['판매수량'].sum().reset_index()
ramen_data = grouped_data[grouped_data['대분류'] == '면류.라면류']
pivot_data = grouped_data.pivot_table(index=['연도', '월'], columns='대분류', values='판매수량', fill_value=0)
corr_matrix = pivot_data.corr()
ramen_corr = corr_matrix['면류.라면류'].drop('면류.라면류').sort_values(ascending=False)
print(ramen_corr)

##########################################################################################################################################################
#데이터 전처리
data=combined_data.copy()
data = data.loc[data['대분류'].isin(['면류.라면류', '생활잡화', '가공식품류'])].copy()
data.drop('중분류',axis=1,inplace=True)
data['년']=data['판매일'].dt.year
data['월']=data['판매일'].dt.month
data.drop('판매일',axis=1,inplace=True)
data.drop('소분류',axis=1,inplace=True)
data = data[data['구분'] == '매출']
data.drop('구분',axis=1,inplace=True)
data.drop('우편번호',axis=1,inplace=True)
data.drop('매출처코드',axis=1,inplace=True)
data.drop('상품 바코드(대한상의)',axis=1,inplace=True)
data.drop('상품명',axis=1,inplace=True)
data.drop('옵션코드',axis=1,inplace=True)
data.drop('규격',axis=1,inplace=True)
data.drop('입수',axis=1,inplace=True)
data.drop('Unnamed: 13',axis=1,inplace=True)
data.reset_index(drop=True,inplace=True)

#데이터 프레임 구조 변경
gd=data.groupby(['대분류','년','월'])['판매수량'].sum().reset_index()
pivot_table = gd.pivot_table(index=['년', '월'], columns='대분류', values='판매수량', fill_value=0)
pivot_table = pivot_table.fillna(0)
pivot_table = pivot_table[['면류.라면류', '생활잡화', '가공식품류']]
pivot_table.reset_index(inplace=True)

#시각화
pivot_table['년월'] = pivot_table['년'].astype(str) + '-' + pivot_table['월'].astype(str)
plt.figure(figsize=(10, 6))
plt.plot(pivot_table['년월'], pivot_table['면류.라면류'], label='면류.라면류')
plt.plot(pivot_table['년월'], pivot_table['생활잡화'], label='생활잡화')
plt.plot(pivot_table['년월'], pivot_table['가공식품류'], label='가공식품류')
plt.title('대분류별 월간 판매수량 추이')
plt.xlabel('년-월')
plt.ylabel('판매수량 합계')
plt.xticks(rotation=45)
plt.legend()
plt.savefig(save_path+'3kind.png')
plt.close()

##########################################################################################################################################################

#일별 데이터 합계
data4=combined_data.copy()
data4 = data4.loc[data4['대분류'].isin(['면류.라면류'])].copy()
data4['판매년']=data4['판매일'].dt.year
data_2021 = data4[data4['판매년']== 2021]
data_2022 = data4[data4['판매년'] == 2022]
data_2023 = data4[data4['판매년'] == 2023]
d2021=data_2021.groupby('판매일')['판매수량'].sum().reset_index()
d2022=data_2022.groupby('판매일')['판매수량'].sum().reset_index()
d2023=data_2023.groupby('판매일')['판매수량'].sum().reset_index()
plt.figure(figsize=(12, 6))  # 그래프 크기 확대
plt.plot(d2021['판매일'], d2021['판매수량'], color='blue', linestyle='-', linewidth=2, marker='o', label='2021 판매수량')
plt.title('2021 면류.라면류 일별 판매수량', fontsize=16, fontweight='bold')
plt.xlabel('날짜', fontsize=12)
plt.ylabel('판매수량', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(save_path+'2021.png')
plt.close()
plt.figure(figsize=(12, 6))  # 그래프 크기 확대
plt.plot(d2022['판매일'], d2022['판매수량'], color='blue', linestyle='-', linewidth=2, marker='o', label='2022 판매수량')
plt.title('2022 면류.라면류 일별 판매수량', fontsize=16, fontweight='bold')
plt.xlabel('날짜', fontsize=12)
plt.ylabel('판매수량', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(save_path+'2022.png')
plt.close()
plt.figure(figsize=(12, 6))  # 그래프 크기 확대
plt.plot(d2023['판매일'], d2023['판매수량'], color='blue', linestyle='-', linewidth=2, marker='o', label='2023 판매수량')
plt.title('2023 면류.라면류 일별 판매수량', fontsize=16, fontweight='bold')
plt.xlabel('날짜', fontsize=12)
plt.ylabel('판매수량', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.tight_layout()
plt.savefig(save_path+'2023.png')
plt.close()
##########################################################################################################################################################
def count_holidays_and_weekends(year, month,korean_holidays=korean_holidays):
    start_date = pd.Timestamp(year, month, 1)
    end_date = start_date + pd.offsets.MonthEnd(1)
    date_range = pd.date_range(start=start_date, end=end_date)
    new_data = pd.DataFrame({'판매일': date_range})
    new_data['휴일여부'] = new_data['판매일'].dt.day_name().isin(['Saturday', 'Sunday']) | new_data['판매일'].isin(korean_holidays)
    return new_data['휴일여부'].sum()

pivot_table_c=pivot_table.copy()
holiday_counts=[]
for year in [2021,2022,2023]:
    for month in range(1, 13):
        holiday_count = count_holidays_and_weekends(year, month)
        holiday_counts.append({'년': year, '월': month, '휴일 개수': holiday_count})
holiday_df = pd.DataFrame(holiday_counts)
pivot_table_c = pd.merge(pivot_table_c, holiday_df, on=['년', '월'], how='left')
##########################################################################################################################################################
correlation_matrix = pivot_table_c[['면류.라면류', '생활잡화', '가공식품류', '휴일 개수']].corr()
ramen_corr = correlation_matrix['면류.라면류'].drop('면류.라면류').abs()
plt.figure(figsize=(8, 6))
ramen_corr.plot(kind='bar', color='skyblue')
plt.title('Absolute Correlation with 면류.라면류', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Absolute Correlation Coefficient', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
for i, v in enumerate(ramen_corr):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(save_path+'corr.png')
plt.close()
##########################################################################################################################################################

from prophet import Prophet
df_prophet = pivot_table_c[['년', '월', '면류.라면류', '휴일 개수']].copy()
df_prophet['ds'] = pd.to_datetime(df_prophet['년'].astype(str) + '-' + df_prophet['월'].astype(str) + '-01')
df_prophet = df_prophet[['ds', '면류.라면류', '휴일 개수']]

df_prophet.to_excel(save_path+'df_prophet.xlsx', index=False)


df_prophet.rename(columns={'면류.라면류': 'y'}, inplace=True)
model = Prophet(yearly_seasonality=True)
model.add_regressor('휴일 개수')
model.fit(df_prophet)
future_dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='MS')
future = pd.DataFrame({'ds': future_dates})
holiday_counts=[]
for year in [2024]:
    for month in range(1, 7):
        holiday_count = count_holidays_and_weekends(year, month)
        holiday_counts.append(holiday_count)

future['휴일 개수'] = holiday_counts
forecast = model.predict(future)
plt.figure(figsize=(10, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='예측값', linestyle='-', marker='o')
plt.title('2024년 1~6월 면류.라면류 판매수량 예측')
plt.xlabel('날짜')
plt.ylabel('예측 판매수량')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path + 'prophet_prediction_fixed.png')
plt.close()
print(forecast[['ds', 'yhat']])


data_2021 = pivot_table_c.loc[pivot_table_c['년']== 2021,['월','면류.라면류']].copy()
data_2022 = pivot_table_c.loc[pivot_table_c['년'] == 2022,['월','면류.라면류']].copy()
data_2023 = pivot_table_c.loc[pivot_table_c['년'] == 2023,['월','면류.라면류']].copy()
plt.figure(figsize=(12, 6))
plt.plot(data_2021['월'], data_2021['면류.라면류'], color='green', label='2021')
plt.plot(data_2022['월'], data_2022['면류.라면류'], color='black', label='2022')
plt.plot(data_2023['월'], data_2023['면류.라면류'], color='blue', label='2023')
plt.plot(range(1,7), forecast['yhat'], color='red', label='2024')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Sales by Month for 2021, 2022, 2023, 2024')
plt.xticks(range(1, 13))  # X축에 1월~12월까지 표시
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_path + f'prophet_result.png')
plt.close()
##########################################################################################################################################################
forecast['yhat'].to_excel(save_path+'result.xlsx', index=False)

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
model.plot_components(forecast)
plt.savefig(save_path + 'prophet_components.png')
plt.close()


train_forecast = model.predict(df_prophet)
model.plot_components(train_forecast)
plt.savefig(save_path + 'train_components.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
plt.plot(forecast['ds'], forecast['yhat'], label='예측값', linestyle='-',color='red', marker='o')
plt.title('2024년 1~6월 면류.라면류 판매수량 예측 (신뢰 구간 포함)')
plt.xlabel('날짜')
plt.ylabel('예측 판매수량')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path + 'prophet_prediction_with_uncertainty.png')
plt.close()