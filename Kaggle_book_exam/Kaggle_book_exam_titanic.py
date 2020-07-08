
#%%
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
# %%
class MyDummyClassifier(BaseEstimator):
    # fit() 메서드는 아무것도 학습하지 않음.
    def fit(self, x, y=None):
        pass

    # predict() 메서드는 단순히 Sex 피처가 1이면 0 그렇지 않으면 1
    def predict(self, x):
        pred = np.zeros( (x.shape[0],1))

        for i in range (x.shape[0]):
            if x['Sex'].iloc[i] == 1:
                pred[i] = 0
            else :
                pred[i] = 1
        
        return pred
#%%

## Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

## 머신러닝에 불필요한 피처 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

## Label Encoding 수행
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

#%%
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df
#%%
Data_file_path='../../DataSet/titanic/'
titanic_df = pd.read_csv(Data_file_path+'train.csv')

#%%
# 데이터 정제
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis =1)
x_titanic_df = transform_features(x_titanic_df)
x_train, x_test, y_train, y_test = train_test_split(x_titanic_df, y_titanic_df,
                                                    test_size=0.2, random_state=0)
#%%
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#%%
myclf = MyDummyClassifier()
myclf.fit(x_train,y_train)

# %%
mypredictions = myclf.predict(x_test)
print('Dummy classifier의 정확도는 : {0:.4f}'.format(accuracy_score(y_test, mypredictions)))

# %%
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test,pred)

    print('오차 행렬')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}'.format(accuracy,precision, recall))


# %%
x_train, x_test, y_train, y_test = train_test_split(x_titanic_df, y_titanic_df,
                                                    test_size=0.20, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
get_clf_eval(y_test,pred)


# %%
pred_proba = lr_clf.predict_proba(x_test)
pred = lr_clf.predict(x_test)
print('pred_proba() 결과 Shape : {0}'.format(pred_proba.shape))
print('pred_proba array에서 앞 3개만 샘플로 추출 \n :',pred_proba[:3])


# %%
