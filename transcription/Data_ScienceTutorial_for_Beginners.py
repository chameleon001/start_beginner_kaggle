
'''
이 글은 아래  'Data ScienceTutorial for Beginners'를 번역한 글 입니다.
최대한 부드럽게 번역하려 노력했으며, 원글의 표현에서 벗어나지 않는 선에서 초월번역(?)을 했습니다.
원글은 아래 링크를 통해 확인하실 수 있습니다.
영어가 필요하다 싶은 부분들은 원문 그대로 썼으며, 많이 부족하다보니 언제든지 피드백 환영합니다.

함수 파라미터 관련은 번역이 필요하면 번역합니다.
'''
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/

# Data ScienceTutorial for Beginners

'''
DATA SCIENTIST

이 튜토리얼은 Data Scientist가 되기 위해 필요한 것을 설명하기 위해 작성되었다.

Data Scientist는 아래의 기술들이 필요하다.
    1. Basic Tools(기본 툴) : Python, R or SQL. 물론 모든것을 알 필요는 없다. 파이썬만 사용할 줄 알아도 충분하다.
    2. Basic Statistics(기본 통계): 평균, 표준편차, 중간값. 기본 통계들을 알고 있다면 파이썬을 쉽게 사용할 수 있다.
    3. Data Munging(데이터 정제) : 정리가 되지 않은 데이터. 예를 들어 일치하지 않는 날짜, 문자열 형식 ex) 2020-01-01 , 01-02
                                Python은 당신의 생각대로 정리가 되도록 도와준다.
    4. Data Visualization(데이터 시각화) : Data Visualization은 설명과 같다. matplot, seaborn 같은 라이브러리를 이용하여 파이썬으로 데이터를 시각화한다.
    5. Machine Learning(기계 학습) : Machine Learning 기술에 있는 수학을 이해할 필요는 없다. 기계학습의 기본을 이해하고 파이썬을 사용하여 구현하는 방법을 배우면 된다.

    요약을 하자면 Data Scientist가 되는 파이썬을 배울 것이다.
    
목차
    1.Introduction to Python:

        Matplotlib
        Dictionaries
        Pandas
        Logic, control flow and filtering
        Loop data structures

    2.Python Data Science Toolbox:

        User defined function
        Scope
        Nested function
        Default and flexible arguments
        Lambda function
        Anonymous function
        Iterators
        List comprehension

    3.Cleaning Data

        Diagnose data for cleaning
        Exploratory data analysis
        Visual exploratory data analysis
        Tidy data
        Pivoting data
        Concatenating data
        Data types
        Missing data and testing with assert

    4.Pandas Foundation

        Review of pandas
        Building data frames from scratch
        Visual exploratory data analysis
        Statistical explatory data analysis
        Indexing pandas time series
        Resampling pandas time series

    5.Manipulating Data Frames with Pandas

        Indexing data frames
        Slicing data frames
        Filtering data frames
        Transforming data frames
        Index objects and labeled data
        Hierarchical indexing
        Pivoting data frames
        Stacking and unstacking data frames
        Melting data frames
        Categoricals and groupby
    6.Data Visualization

        Seaborn: https://www.kaggle.com/kanncaa1/seaborn-for-beginners
        Bokeh 1: https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-1
        Rare Visualization: https://www.kaggle.com/kanncaa1/rare-visualization-tools
        Plotly: https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners

    7.Machine Learning

        https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners/

    8.Deep Learning

        https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners

    9.Time Series Prediction

        https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda

    10.Statistic

        https://www.kaggle.com/kanncaa1/basic-statistic-tutorial-for-beginners

    11.Deep Learning with Pytorch

        Artificial Neural Network: https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
        Convolutional Neural Network: https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
        Recurrent Neural Network: https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch

'''

#%%
'''
    이 Python3 환경에는 유용한 분석 라이브러리들이 많이 설치 되어있습니다.
    kaggle/python docker 이미지는 https://github.com/kaggle/docker-python
    에서 받아보실수 있습니다.
    예를 들어 불러올 때 유용한 패키지들이 몇개 있습니다.
'''

import numpy as np # Linear Algebra (선형대수)
import pandas as pd # Data Processing CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# 입력 데이터는 "../input/" 폴더 안에 있습니다. 
# 예를 들어,  실행하면(실행을 클릭하거나 Shift+Enter를 눌러) 입력 폴더에 파일이 나열됩니다.

# *역자는 파일을 다운받아 로컬에서 진행하기 때문에 file_path를 따로 써주었다.
# 원코드는 주석으로 쳐두었습니다. 캐글 내에서 진행하시는 분들은 원본 코드로 진행 해주시기 바랍니다
file_path = '../../DataSet/2619_4359_bundle_archive'

#%%

from subprocess import check_output
#원본
#print(check_output(["ls", "../input"]).decode("utf8"))
#역자
print(check_output(["ls", file_path]).decode("utf8"))

import os
#원본
#for dirname, _, filenames in os.walk("/kaggle/input"):
#역자
for dirname, _, filenames in os.walk(file_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 경고.
#   위의 코드를 실행할 때에 그림과 같이 경고 메세지가 뜬다면 pd.read_csv()에 그림과 같이 ".csv" 경로를 넣어주어야 한다.
# %%
#data = pd.read_csv('/kaggle/input/pokemon.csv')
#data.head()

#원본
# data = pd.read_csv('../input/pokemon.csv')
#역자
data = pd.read_csv(file_path+'/pokemon.csv')

data.head()

data.info()

data.corr()
# %%

# correlation map (상관 관계)
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

data.head(10)

data.columns
# %%

'''
1. INTRODUCTION TO PYTHON

    MATPLOTLIB
        Matplot은 Data를 그리는데에 도움을 주는 python 라이브러리입니다.
        쉽고 기본적인 plot은 선(Line), 분산(Scatter), 히스토그램(Histogram) Plot입니다.
            Line plot은 X축이 시간일때 더 좋습니다.
            Scatter는 두 변수 사이에 상관 관계가 있을 때 사용하면 좋습니다.
            Histogram은 수치 데이터의 분포를 볼 때 좋습니다.
            Customization : 색상, 레이블, 선 두께, 불 투명도, 격자, 그림크기, 축 눈금, 선 스타일
            (Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle)

'''


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

# %%
# Scatter Plot
# x = attack, y = defense

data.plot(kind = 'scatter', x='Attack', y='Defense', alpha = 0.5, color = 'red')
plt.xlabel('Attack')    #label = name of label 
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot') #title = title of plot


# %%
# Histogram
# bins = number of bar in figure

data.Speed.plot(kind = 'hist', bins = 50, figsize = (12,12))
plt.show()

# %%
# clf() : 정리하면 다시 새롭게 만들 수 있습니다.

data.Speed.plot(kind = 'hist', bins =50)
plt.clf()
#우리는 clf()함수 때문에 plot을 볼 수 없습니다.
plt.show()


# %%

'''
DICTIONARY      *역자 : python의 자료형 중 하나입니다.
    왜 우리가 dictionary가 필요한가?
        - 'Key' 와 'Value'를 갖는다.
        - lists 보다 빠르다.
        - key와 value가 무엇인가. ex) dictionary = {'spain' :'madrid'}
          key는 'spain' , values는 'madrid' 이다.
    
        Dictionary는 쉽다.
        keys (), values ​​(), update, add, check, remove key 등을 연습해보자.
    
'''
#%%

# Dictionary를 만들고, Key와 Values를 확인해보자.

dictionary = {'spain' : 'madrid', 'usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())

# %%


# Key는 string, boolean, float, integer, tubles와 같은 불변의 객체이어야 한다. 
#   *역자 : Dictionary의 key는 값을 변경할 수 없는. Immutable 타입이어야하며, Value는 Immutable과 mutable 모두 가능하다.
#   *       key로 문자열이나 tuble은 사용될 수 있으나 list는 key로 사용될 수는 없다.
# List는 불변하지 않다.
# Keys는 특별해야한다.(고유한 값이다.)
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)

# %%
'''
PANDAS
    pandas에 대해 알아야 할 것은 무엇인가?
        -CSV : comma -구분된 값   *역자 : comma(,)로 구분되어진 값 ex) A,B,C
'''

#data = pd.read_csv('../input/pokemon.csv')
data = pd.read_csv(file_path+'/pokemon.csv')

series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))
# %%
'''
pandas를 계속하기 전에, 논리(Logic), 제어 흐름(control flow), 필터링(filtering)을 배워야 한다.
비교 연산자 : ==, <, >, <=
Boolean 연산자: and, or ,not
Filtering pandas
'''
# 비교 연산자
print(3>2)
print(3!=2)
# Boolean 연산자
print(True and False)
print(True or False)

# 1 - Filtering Pandas data frame
x = data['Defense']>200    # 방어력이 200보다 높은 포켓몬은 3마리 입니다.
data[x]

# 2 - F
# %%
