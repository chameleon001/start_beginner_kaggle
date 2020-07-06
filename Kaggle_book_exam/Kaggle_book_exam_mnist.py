
#%%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#%%

class MyFakeClassifier(BaseEstimator):

    def fit(self, X, y):
        pass

    # 입력값으로 들어오는 x 데이터 세트의 크기만큼 모두 0값으로 만들어서 반환
    def predict(self, X):
        return np.zeros( (len(X), 1), dtype=bool)

#%%
digits = load_digits()

y = (digits.target ==7).astype(int)
x_train, x_test, y_train, y_test = train_test_split(digits.data, y,random_state=11)

#%%
print('레이블 테스트 세트 크기 :', y_test.shape)
print('테스트 세트 레이블 0 과 1의 분포도')
print(pd.Series(y_test).value_counts())

#%%
# Dummy Classifier로 학습/예측/정확도 평가.
fakeclf = MyFakeClassifier()
fakeclf.fit(x_train, y_train)
fakepred = fakeclf.predict(x_test)
print('모든 예측을 0으로 하여도 정확도는 :{:.3f}'.format(accuracy_score(y_test,fakepred)))

# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, fakepred)


# %%
