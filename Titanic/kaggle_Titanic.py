
#%%

# Titanic - 데이터

# Passengerid : 탑승자 데이터 일련번호
# survived : 생존 여부, 0 = 사망, 1 = 생존
# pclass : 티켓의 선실 등급, 1 = 일등석, 2 = 이동석, 3 = 삼등석
# sex : 탑승자 성별
# name : 탑승자 이름
# Age : 나이
# sibsp : 동승자 인원수(형제자매,배우자)
# parch : 동승자 인원수 (부모님,아이)
# ticket : 티켓번호
# fare : 요금
# cabin : 선실 번호
# embarked : 경유지 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

titanic_df = pd.read_csv('D:/github/\kaggle_dataset/titanic/train.csv')
titanic_df.head(3)

# %%

print('\n ## 학습 데이터 ## \n')
print(titanic_df.info())

# %%

# Null 값들을 평균이나 고정값으로 변경.
print('데이터 세트 Null 값 개수 ', titanic_df.isnull().sum().sum())

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace = True)
titanic_df['Cabin'].fillna('N', inplace = True)
titanic_df['Embarked'].fillna('N', inplace = True)

print('데이터 세트 Null 값 개수 ', titanic_df.isnull().sum().sum())

# %%
print(' Sex 값 분포 ')