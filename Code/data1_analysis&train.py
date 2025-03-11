
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', None)

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
    '2023-12-25'  # 성탄절
])


pd.set_option('display.max_columns', None)
path = 'C:/Users/User/Desktop/Project/유통/수요예측/중소유통물류센터 거래 데이터/'
all_files = os.listdir(path)
xlsx_files = [f for f in all_files if f.endswith('.xlsx')]

save_path='C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/Try7/'

all_sheets = pd.read_excel(os.path.join(path, xlsx_files[0]), sheet_name=None)
combined_data = pd.concat(all_sheets.values(), ignore_index=True)
data=combined_data.copy()


##########################################################################################
data.drop('대분류',axis=1,inplace=True)
data=data.loc[data['중분류']=='라면,통조림,상온즉석']
data.drop('중분류',axis=1,inplace=True)
data.drop('소분류',axis=1,inplace=True)
data['쉬는날여부'] = data['판매일'].dt.day_name().isin(['Saturday', 'Sunday']) | data['판매일'].isin(korean_holidays)
data['쉬는날여부'] = data['쉬는날여부'].astype(int)
data['요일'] = data['판매일'].dt.day_name()
data = data.loc[data['구분'] == '매출'].copy()
data.drop('구분',axis=1,inplace=True)
data.drop('우편번호',axis=1,inplace=True)
data.drop('상품 바코드(대한상의)',axis=1,inplace=True)
data.drop('상품명',axis=1,inplace=True)
data.drop('옵션코드',axis=1,inplace=True)
data.drop('규격',axis=1,inplace=True)
data.drop('입수',axis=1,inplace=True)
data['판매년'] = data['판매일'].dt.year
data.reset_index(drop=True,inplace=True)
data_copy=data.copy()
data.drop('판매년',axis=1,inplace=True)
##########################################################################################
gd= data.groupby('판매일')['판매수량'].sum().reset_index()
data1_unique = data.drop_duplicates(subset=['판매일'])[['판매일', '쉬는날여부', '요일']]
gd = pd.merge(gd, data1_unique, on='판매일', how='left')
##########################################################################################
all_dates = pd.date_range(start='2021-01-01', end='2023-12-31')
gd['판매일'] = pd.to_datetime(gd['판매일'])
missing_dates = all_dates.difference(gd['판매일'])
missing_data = pd.DataFrame({'판매일': missing_dates,'판매수량': 0})
missing_data['쉬는날여부'] = missing_data['판매일'].dt.day_name().isin(['Saturday', 'Sunday']) | missing_data['판매일'].isin(korean_holidays)
missing_data['쉬는날여부'] = missing_data['쉬는날여부'].astype(int)
missing_data['요일'] = missing_data['판매일'].dt.day_name()
gd_filled = pd.concat([gd, missing_data]).sort_values(by='판매일').reset_index(drop=True).copy()
##########################################################################################
holidays_extended = []
for holiday in korean_holidays:
    for offset in range(-1, 2):
        holidays_extended.append(holiday + pd.DateOffset(days=offset))
gd_filled['연휴전후'] = gd_filled['판매일'].isin(holidays_extended) | gd_filled['판매일'].dt.day_name().isin(['Monday', 'Friday'])
gd_filled['연휴전후'] = gd_filled['연휴전후'].astype(bool).astype(int)
gd_filled.loc[gd_filled['판매수량'] == 0, '연휴전후'] = 0    #판매수량이 0 -> 장사 안함  !=  연휴전후는 판매수량 큰날을 표시하기 위함
gd_filled.loc[gd_filled['쉬는날여부'] == 1, '연휴전후'] = 0 # 판매수량 적음 !=  연휴전후는 판매수량 큰날을 표시하기 위함
#########################################################################################################################################################

y=gd_filled['판매수량']
x=gd_filled.drop('판매수량',axis=1)
x = pd.get_dummies(x, columns=['요일'], drop_first=True)
x['판매월'] = x['판매일'].dt.month
x['판매주차'] = x['판매일'].dt.strftime('%W').astype(int)
x['판매일_일자'] = x['판매일'].dt.day
x.drop('판매일', axis=1, inplace=True)
x = x.astype(int)
x.to_excel('C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/x.xlsx', index=False)

y.to_excel('C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/y.xlsx', index=False)
#########################################################################################################################################################
h=gd_filled.loc[gd_filled['쉬는날여부']==1]
hn=gd_filled.loc[gd_filled['연휴전후']==1]
nh=gd_filled.loc[-((gd_filled['쉬는날여부']==1)|(gd_filled['연휴전후']==1))]

holiday_mean=h['판매수량'].mean()
holiday_after_before_mean=hn['판매수량'].mean()
not_holiday_mean=nh['판매수량'].mean()


categories = ['Holiday', 'Holiday Before/After', 'Others']
values = [holiday_mean, holiday_after_before_mean, not_holiday_mean]
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['#FF9999', '#66B2FF', '#99FF99'], edgecolor='black')
plt.title('Average Sales Quantity by Day Type', fontsize=16)
plt.xlabel('Day Type', fontsize=14)
plt.ylabel('Average Sales Quantity', fontsize=14)
for i, v in enumerate(values):
    plt.text(i, v + 2, f'{v:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('C:/Users/User/Desktop/Project/유통/수요예측/actual_vs_predicted_after_gridsearch.png')
plt.close()
#########################################################################################################################################################


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#########################################################################################################################################################

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": GradientBoostingRegressor(random_state=42)  # Placeholder for LightGBM
}

# models = {
#     "DecisionTree": DecisionTreeRegressor(random_state=42),
#     "RandomForest": RandomForestRegressor(random_state=3102,n_estimators=400,max_depth=6,min_samples_split=22, min_samples_leaf=4),
#     "XGBoost": XGBRegressor(random_state=42),
#     "LightGBM": GradientBoostingRegressor(random_state=42)  # Placeholder for LightGBM
# }
# 각 모델에 대해 교차 검증 수행
cv_scores = {}
for name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=10)
    cv_scores[name] = cv_score.mean()
# 교차 검증 평균 점수를 가로 바 그래프로 시각화
model_names = list(cv_scores.keys())
scores = list(cv_scores.values())

plt.figure(figsize=(10, 6))
bars = plt.barh(model_names, scores, color='skyblue')
plt.xlabel('Cross-Validation Score')
plt.title('Comparison of Tree-Based Models')
plt.xlim(min(scores) - 0.01, max(scores) + 0.01)
plt.grid(True)
# 각 바 위에 값 표시
for bar in bars:
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center')
plt.savefig('C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/model_selsect_real.png')
plt.close()



save_path='C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/Try8/'
#########################################################################################################################################################

rf = RandomForestRegressor(random_state=42)
# 하이퍼파라미터 범위 설정
param_grid = {
    'n_estimators': [400],
    'max_depth': [6],
    'min_samples_split': [22],
    'min_samples_leaf': [4],
    'random_state': range(102, 5003, 100),
}
# 랜덤 포레스트 모델 생성
rf = RandomForestRegressor(random_state=42)
# GridSearchCV 설정 (5-fold 교차 검증 사용)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# 모델 학습
grid_search.fit(X_train, y_train)
# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)

# 최적의 모델로 예측
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 모델 평가 (MSE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 결과 출력
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape * 100}%')

# 결과를 txt 파일로 저장
with open(save_path + 'regression_evaluation.txt', 'w') as file:
    file.write(f'Mean Squared Error (MSE): {mse}\n')
    file.write(f'Mean Absolute Error (MAE): {mae}\n')
    file.write(f'R-squared (R2): {r2}\n')
    file.write(f'Mean Absolute Percentage Error (MAPE): {mape * 100}%\n')


# 예측값과 실제값 비교 그래프
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values', color='blue', linewidth=2)
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Sales after GridSearchCV')
plt.xlabel('Samples')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path + 'actual_vs_predicted_after_gridsearch.png')
plt.close()



date_range = pd.date_range(start="2024-01-01", end="2024-06-30")
korean_holidays_2024 = pd.to_datetime([
    '2024-01-01',  # 신정
    '2024-02-09', '2024-02-10', '2024-02-11',  # 설날 연휴
    '2024-02-12',  # 설날 대체 공휴일 (일요일에 겹침)
    '2024-03-01',  # 삼일절
    '2024-04-10',  # 총선
    '2024-05-05',  # 어린이날
    '2024-05-06',  # 어린이날 대체 공휴일 (일요일에 겹침)
    '2024-05-15',  # 석가탄신일
    '2024-06-06',  # 현충일
])
new_data = pd.DataFrame({'판매일': date_range})
new_data['쉬는날여부'] = new_data['판매일'].dt.day_name().isin(['Saturday', 'Sunday']) | new_data['판매일'].isin(korean_holidays_2024)
new_data['요일'] = new_data['판매일'].dt.day_name()
new_data['판매월'] = new_data['판매일'].dt.month
new_data['판매주차'] = new_data['판매일'].dt.isocalendar().week
new_data['판매일_일자'] = new_data['판매일'].dt.day
holidays_extended = []
for holiday in korean_holidays_2024:
    for offset in range(-2, 3):  # 연휴 전후 3일 포함
        holidays_extended.append(holiday + pd.DateOffset(days=offset))
new_data['연휴전후'] = new_data['판매일'].isin(holidays_extended).astype(int)
new_data = pd.get_dummies(new_data, columns=['요일'])
new_data.drop('판매일', axis=1, inplace=True)
x_2024 = new_data.astype(int)
x_2024 = x_2024[x.columns]
print(x_2024.head())


y_2024_pred = best_rf.predict(x_2024)

y_2024_pred_df = pd.DataFrame(y_2024_pred, columns=['Predicted Sales'])

combined_df = pd.concat([x_2024, y_2024_pred_df], axis=1)

result = combined_df.groupby('판매월')['Predicted Sales'].sum()




# 연도별 데이터 분리
data_2021 = data_copy.loc[data_copy['판매년'] == 2021].copy()
data_2022 = data_copy.loc[data_copy['판매년'] == 2022].copy()
data_2023 = data_copy.loc[data_copy['판매년'] == 2023].copy()
data_2021['월'] = data_2021['판매일'].dt.month
data_2022['월'] = data_2022['판매일'].dt.month
data_2023['월'] = data_2023['판매일'].dt.month
data_2021_sum = data_2021.groupby('월')['판매수량'].sum().reset_index()
data_2022_sum = data_2022.groupby('월')['판매수량'].sum().reset_index()
data_2023_sum = data_2023.groupby('월')['판매수량'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(data_2021_sum['월'], data_2021_sum['판매수량'], color='red', label='2021')
plt.plot(data_2022_sum['월'], data_2022_sum['판매수량'], color='black', label='2022')
plt.plot(data_2023_sum['월'], data_2023_sum['판매수량'], color='blue', label='2023')
plt.plot(range(1,7), result, color='green', label='2024')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Sales by Month for 2021, 2022, 2023')
plt.xticks(range(1, 13))  # X축에 1월~12월까지 표시
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_path + 'result.png')
plt.close()

import joblib
joblib.dump(best_rf, save_path + 'random_forest_model.joblib')


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (Windows의 경우)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 맑은 고딕 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
# 데이터프레임 만들기
df = pd.DataFrame()
df['feature'] = list(x)
df['importance'] = best_rf.feature_importances_
df.sort_values(by='importance', ascending=True, inplace=True)
# 시각화
plt.figure(figsize=(5, 5))
bars = plt.barh(df['feature'], df['importance'])
for bar in bars:
    plt.text(
        bar.get_width(),  # 바의 끝부분에 위치
        bar.get_y() + bar.get_height() / 2,  # 바의 가운데에 위치
        f'{bar.get_width():.4f}',  # 중요도 값을 표시
        va='center'  # 텍스트의 수직 정렬
    )
plt.tight_layout()
plt.savefig(save_path + 'feature_importance.png')
plt.close()


# 모델 불러오기
#loaded_rf = joblib.load('random_forest_model.joblib')


save_path='C:/Users/User/Desktop/Project/유통/수요예측/1 분석결과/Try13/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
#########################################################################################################################################################
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# LightGBM 모델 생성
lgbm = LGBMRegressor(random_state=42)

# 하이퍼파라미터 범위 설정
param_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
'max_depth': [1, 2, 3, 4, 5],
'min_child_samples': [16, 18, 20, 22, 24],
'num_leaves': [20, 25, 31, 35, 40],
'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
'random_state': [42],
}

# GridSearchCV 설정 (5-fold 교차 검증 사용)
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)



# 최적의 모델로 예측
best_lgbm = grid_search.best_estimator_
y_pred = best_lgbm.predict(X_test)

# 모델 평가 (MSE, MAE, R-squared, MAPE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 결과 출력
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape * 100}%')

# 결과를 txt 파일로 저장
with open(save_path + 'regression_evaluation.txt', 'w') as file:
    file.write(f'Mean Squared Error (MSE): {mse}\n')
    file.write(f'Mean Absolute Error (MAE): {mae}\n')
    file.write(f'R-squared (R2): {r2}\n')
    file.write(f'Mean Absolute Percentage Error (MAPE): {mape * 100}%\n')



import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (Windows의 경우)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 맑은 고딕 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 예측값과 실제값 비교 그래프
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values', color='blue', linewidth=2)
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Sales after GridSearchCV')
plt.xlabel('Samples')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path + 'actual_vs_predicted_after_gridsearch.png')
plt.close()



date_range = pd.date_range(start="2024-01-01", end="2024-06-30")
korean_holidays_2024 = pd.to_datetime([
    '2024-01-01',  # 신정
    '2024-02-09', '2024-02-10', '2024-02-11',  # 설날 연휴
    '2024-02-12',  # 설날 대체 공휴일 (일요일에 겹침)
    '2024-03-01',  # 삼일절
    '2024-04-10',  # 총선
    '2024-05-05',  # 어린이날
    '2024-05-06',  # 어린이날 대체 공휴일 (일요일에 겹침)
    '2024-05-15',  # 석가탄신일
    '2024-06-06',  # 현충일
])
new_data = pd.DataFrame({'판매일': date_range})
new_data['쉬는날여부'] = new_data['판매일'].dt.day_name().isin(['Saturday', 'Sunday']) | new_data['판매일'].isin(korean_holidays_2024)
new_data['요일'] = new_data['판매일'].dt.day_name()
new_data['판매월'] = new_data['판매일'].dt.month
new_data['판매주차'] = new_data['판매일'].dt.isocalendar().week
new_data['판매일_일자'] = new_data['판매일'].dt.day
holidays_extended = []
for holiday in korean_holidays_2024:
    for offset in range(-2, 3):  # 연휴 전후 3일 포함
        holidays_extended.append(holiday + pd.DateOffset(days=offset))
new_data['연휴전후'] = new_data['판매일'].isin(holidays_extended).astype(int)
new_data = pd.get_dummies(new_data, columns=['요일'])
new_data.drop('판매일', axis=1, inplace=True)
x_2024 = new_data.astype(int)
x_2024 = x_2024[x.columns]
print(x_2024.head())


y_2024_pred = best_lgbm.predict(x_2024)

y_2024_pred_df = pd.DataFrame(y_2024_pred, columns=['Predicted Sales'])

combined_df = pd.concat([x_2024, y_2024_pred_df], axis=1)

result = combined_df.groupby('판매월')['Predicted Sales'].sum()




# 연도별 데이터 분리
data_2021 = data_copy.loc[data_copy['판매년'] == 2021].copy()
data_2022 = data_copy.loc[data_copy['판매년'] == 2022].copy()
data_2023 = data_copy.loc[data_copy['판매년'] == 2023].copy()
data_2021['월'] = data_2021['판매일'].dt.month
data_2022['월'] = data_2022['판매일'].dt.month
data_2023['월'] = data_2023['판매일'].dt.month
data_2021_sum = data_2021.groupby('월')['판매수량'].sum().reset_index()
data_2022_sum = data_2022.groupby('월')['판매수량'].sum().reset_index()
data_2023_sum = data_2023.groupby('월')['판매수량'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(data_2021_sum['월'], data_2021_sum['판매수량'], color='green', label='2021')
plt.plot(data_2022_sum['월'], data_2022_sum['판매수량'], color='black', label='2022')
plt.plot(data_2023_sum['월'], data_2023_sum['판매수량'], color='blue', label='2023')
plt.plot(range(1,7), result, color='red', label='2024')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Sales by Month for 2021, 2022, 2023, 2024')
plt.xticks(range(1, 13))  # X축에 1월~12월까지 표시
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_path + 'result.png')
plt.close()
# LightGBM 모델 저장
joblib.dump(best_lgbm, save_path + 'lightgbm_model.joblib')

# 피처 중요도 시각화
df = pd.DataFrame()
df['feature'] = list(X_train.columns)
df['importance'] = best_lgbm.feature_importances_

# 피처 이름 변경
df['feature'] = df['feature'].replace({
    '쉬는날여부': '휴일 여부',
    '연휴전후': '휴일 전/후 여부'
})

df.sort_values(by='importance', ascending=True, inplace=True)

# 시각화
plt.figure(figsize=(5, 5))
bars = plt.barh(df['feature'], df['importance'])
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center')
plt.tight_layout()
plt.savefig(save_path + 'feature_importance.png')
plt.close()

# 모델 불러오기
#import joblib
#loaded_lgbm = joblib.load('lightgbm_model.joblib')



mean_scores = grid_search.cv_results_['mean_test_score']  # 6250개의 점수
mean_scores = np.abs(mean_scores)
mean_scores_per_model = mean_scores.reshape(-1, 10).mean(axis=1)  # 각 모델당 10개의 점수 평균
param_combinations = list(range(1, len(mean_scores_per_model) + 1))  # 1부터 625까지 인덱스
plt.figure(figsize=(10, 6))
plt.plot(param_combinations, mean_scores_per_model, marker='o', linestyle='--', color='b')
plt.xticks([])
plt.title('Model Performance for Different Hyperparameter Combinations')
plt.xlabel('Hyperparameter Combination Index')
plt.ylabel('Mean Test Score (MSE)')
plt.tight_layout()
plt.savefig(save_path + 'gridsearch.png')
plt.close()


best_index = np.argmin(mean_scores_per_model)  # 가장 낮은 MSE 값의 인덱스 (가장 성능이 좋은 모델)
best_score = mean_scores_per_model[best_index]  # 가장 낮은 MSE 값

# 가장 좋은 하이퍼파라미터 조합
best_params = grid_search.cv_results_['params'][best_index]

# 출력
print(f"Best score (MSE): {best_score}")
print(f"Best hyperparameters: {best_params}")



result.to_excel(save_path+'result.xlsx', index=False)