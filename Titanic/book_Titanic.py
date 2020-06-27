
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
Data_file_path='../../DataSet/titanic/'
#%%
titanic_df = pd.read_csv(Data_file_path+'train.csv')
titanic_df.head(3)

# %%
print('\n ### 학습 데이터 ### \n')
print(titanic_df.info())

# %%
# NaN 값을 변경 나이는 mean값 그 외엔 N으로 채워준다.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)


print('DataSet Null 값? : \n',titanic_df.isnull().sum())
print('DataSet Null 값 : ',titanic_df.isnull().sum().sum())

# %%

#Object(string)값들 분류

print('\n Sex 값 분포 : \n', titanic_df['Sex'].value_counts())
print('\n Cabin 값 분포 : \n', titanic_df['Cabin'].value_counts())
print('\n Embarked 값 분포 : \n', titanic_df['Embarked'].value_counts())


# %%
#Cabin 값을 정리하기. 선실 등급별 표시되도록 변경.
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))

#%%
titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()
sns.barplot(x='Sex', y = 'Survived', data=titanic_df)
#%%
sns.barplot(x='Pclass', y='Survived',hue='Sex',data=titanic_df)

#%%
# 나이 대 별로 구간정하기
def get_category(age):
    cat = ''

    if age <= -1:
        cat = 'Unknown'
    elif age <= 5:
        cat = 'Baby'
    elif age <= 12:
        cat = 'Child'
    elif age <= 18:
        cat = 'Teenager'
    elif age <= 25:
        cat = 'Student'
    elif age <= 35:
        cat = 'Young Adult'
    elif age <= 60:
        cat = 'Adult'
    else :
        cat = 'Elderly'

    return cat

# %%
plt.figure(figsize=(10,6))

#%%
group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult', 'Adult','Elderly']

#%%
# lambda 식을 이용하여 나이에 해당하는 Age 칼럼 값 받기
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat',axis=1, inplace=True)

# %%
from sklearn import preprocessing

#%%
def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF


# %%
titanic_df = encode_features(titanic_df)
titanic_df.head()

# %%
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)

    return df

# 불필요한 카테고리 드랍.
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    
    return df

#데이터 전처리 
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


#%%
# 원본 데이터를 재로딩하고, 피처 데이터 세트와 레이블 데이터 세트 추출
titanic_df = pd.read_csv(Data_file_path + 'train.csv')
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived',axis=1)

x_titanic_df = transform_features(x_titanic_df)

#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 결정 트리, 랜덤 포레스트, 로지스틱
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

#%%
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)

print('DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

#%%
rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
print('RandomForestClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# %%
lr_clf.fit(x_train, y_train)
lr_pred = lr_clf.predict(x_test)
print('LogisticRegression 정확도 : {0:.4f}'.format(accuracy_score(y_test, lr_pred)))

# %%
from sklearn.model_selection import KFold

def exec_kfold(clf, fold =5):
    kfold = KFold(n_splits=folds)
    scores = []

