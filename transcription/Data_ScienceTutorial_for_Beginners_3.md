<a id="32"></a> <br>
# Pandas로 데이터 프레임 다루기 (MANIPULATING DATA FRAMES WITH PANDAS)

<a id="33"></a> <br>
### 데이터 프레임 인덱싱(INDEXING DATA FRAMES)
* 대괄호를 사용하여 인덱싱하기.
* 열 속성과 행 레이블 사용하기
* loc 접근자 사용하기  *역자 : loc : 인덱스 기준으로 행 데이터 읽는 함수, iloc : 행번호 기준으로 행데이터 읽기
* 일부 열만 선택하기


```
# read data
data = pd.read_csv(file_path+'pokemon.csv')
data= data.set_index("#")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
# 대괄호 사용하여 인덱싱 하기
data["HP"][1]
```




    45




```
# 열 속성과 행 레이블 사용하기
data.HP[1]
```




    45




```
# loc 함수 사용하기
data.loc[1,["HP"]]
```




    HP    45
    Name: 1, dtype: object




```
# 몇몇 열만 선택하기
data[["HP","Attack"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HP</th>
      <th>Attack</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>796</th>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>797</th>
      <td>50</td>
      <td>160</td>
    </tr>
    <tr>
      <th>798</th>
      <td>80</td>
      <td>110</td>
    </tr>
    <tr>
      <th>799</th>
      <td>80</td>
      <td>160</td>
    </tr>
    <tr>
      <th>800</th>
      <td>80</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 2 columns</p>
</div>



<a id="34"></a> <br>
### 데이터 프레임 자르기(slicing)  (SLICING DATA FRAME)
* 열 선택의 차이점
    * 시리즈(series) 와 데이터 프레임
* 시리즈(series) 인덱싱과 자르기(slicing)
* 역 자르기(slicing)
* 특정 지점부터 끝까지


```
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
```

    <class 'pandas.core.series.Series'>
    <class 'pandas.core.frame.DataFrame'>



```
# 자르기 및 시리즈 인덱싱
자르기 및 시리즈 인덱싱
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive  *역자: 인덱스 번호 1~10까지. HP부터 Defense까지
```


      File "<ipython-input-86-26b3603b7800>", line 2
        자르기 및 시리즈 인덱싱
            ^
    SyntaxError: invalid syntax




```
# 역 슬라이싱
data.loc[10:1:-1,"HP":"Defense"] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>44</td>
      <td>48</td>
      <td>65</td>
    </tr>
    <tr>
      <th>9</th>
      <td>78</td>
      <td>104</td>
      <td>78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78</td>
      <td>84</td>
      <td>78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>58</td>
      <td>64</td>
      <td>58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39</td>
      <td>52</td>
      <td>43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>100</td>
      <td>123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>82</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>62</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>49</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```
# 특정 지점부터 끝까지.
data.loc[1:10,"Speed":] # *역자 : speed부터 끝까지
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



<a id="35"></a> <br>
### 데이터 프레임 필터링   (FILTERING DATA FRAMES)
<br> boolean형 시리즈(series) 만들기
<br> 필터 결합
<br> 다른 열 기반 필터링(Filtering column based others)


```
# boolean형 시리즈(series) 만들기
boolean = data.HP > 200
data[boolean]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122</th>
      <td>Chansey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>250</td>
      <td>5</td>
      <td>5</td>
      <td>35</td>
      <td>105</td>
      <td>50</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>262</th>
      <td>Blissey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>255</td>
      <td>10</td>
      <td>10</td>
      <td>75</td>
      <td>135</td>
      <td>55</td>
      <td>2</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
# 필터 결합
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122</th>
      <td>Chansey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>250</td>
      <td>5</td>
      <td>5</td>
      <td>35</td>
      <td>105</td>
      <td>50</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>262</th>
      <td>Blissey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>255</td>
      <td>10</td>
      <td>10</td>
      <td>75</td>
      <td>135</td>
      <td>55</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>352</th>
      <td>Wailord</td>
      <td>Water</td>
      <td>NaN</td>
      <td>170</td>
      <td>90</td>
      <td>45</td>
      <td>90</td>
      <td>45</td>
      <td>60</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>656</th>
      <td>Alomomola</td>
      <td>Water</td>
      <td>NaN</td>
      <td>165</td>
      <td>75</td>
      <td>80</td>
      <td>40</td>
      <td>45</td>
      <td>65</td>
      <td>5</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
# 다른 열 기반 필터링
data.HP[data.Speed < 15]
```




    #
    231     20
    360     45
    487     50
    496    135
    659     44
    Name: HP, dtype: int64



<a id="36"></a> <br>
### 데이터 변환  (TRANSFORMING DATA)
* 일반 파이썬 함수.
* 람다(Lambda) 함수: 모든 요소에 임의의 파이썬 함수를 적용
* 다른 열을 사용하여 열 정의


```
# 일반 파이썬 함수.
def div(n):
    return n/2
data.HP.apply(div)
```




    #
    1      22.5
    2      30.0
    3      40.0
    4      40.0
    5      19.5
           ... 
    796    25.0
    797    25.0
    798    40.0
    799    40.0
    800    40.0
    Name: HP, Length: 800, dtype: float64




```
# 또는 우리는 람다(lambda) 함수를 사용한다.
data.HP.apply(lambda n : n/2)
```




    #
    1      22.5
    2      30.0
    3      40.0
    4      40.0
    5      19.5
           ... 
    796    25.0
    797    25.0
    798    40.0
    799    40.0
    800    40.0
    Name: HP, Length: 800, dtype: float64




```
#  다른 열을 사용하여 열 정의
data["total_power"] = data.Attack + data.Defense
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>total_power</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>165</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>223</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>



<a id="37"></a> <br>
### 색인 객체(objects)와 라벨링된 데이터.
index: 라벨의 순서




```
# 우리의 인덱스 이름은 이것이다.
print(data.index.name)  #  (#)
# 이것을 바꿔보자
data.index.name = "index_name"
data.head()
```

    #





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>total_power</th>
    </tr>
    <tr>
      <th>index_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>165</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>223</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```
# 인덱스 덮어 쓰기.
# 우리가 인덱스를 수정하기를 원한다면, 우리는 모든 인덱스를 변경해야한다.
data.head()
# data를 data3으로 먼저 복사한 다음에 인덱스를 변경하자.
data3 = data.copy()
# 인덱스를 100에서 시작하도록 합니다. 눈에 띄는 변화는 아니지만 단지 예제일뿐이다.
data3.index = range(100,900,1)
data3.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>total_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
      <td>98</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
      <td>125</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>165</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Mega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>223</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```
# 우리는 열(column)중 하나를 인덱스로 만들 수 있습니다.
# 나는 실제로 pandas section으로 데이터 프레임을 다루기 시작했습니다.
# 이런식이 였다.
# data = data.set_index("#")
# 또한 너는 사용할수 있다.
# data.index = data["#"]
```

<a id="38"></a> <br>
### 계층적 인덱싱  (HIERARCHICAL INDEXING)
* 인덱스 설정


```
# 데이터를 다시 한번 불러 시작해보자.
data = pd.read_csv(file_path + "pokemon.csv")
data.head()
# 보다시피 인덱스가 있다. 그러나 우리는 하나 이상의 열(column)을 인덱스로 설정하기를 원한다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
# 인덱스 설정하기 : type1은 외부 type2는 내부 인덱스이다.
data1 = data.set_index(["Type 1", "Type 2"])
data1.head(100)
# data1.loc["Fire", "Flying"]  # 인덱스 사용법
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
    <tr>
      <th>Type 1</th>
      <th>Type 2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Grass</th>
      <th>Poison</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>4</td>
      <td>Mega Venusaur</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Fire</th>
      <th>NaN</th>
      <td>5</td>
      <td>Charmander</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Poison</th>
      <th>NaN</th>
      <td>96</td>
      <td>Grimer</td>
      <td>80</td>
      <td>80</td>
      <td>50</td>
      <td>40</td>
      <td>50</td>
      <td>25</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>97</td>
      <td>Muk</td>
      <td>105</td>
      <td>105</td>
      <td>75</td>
      <td>65</td>
      <td>100</td>
      <td>50</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Water</th>
      <th>NaN</th>
      <td>98</td>
      <td>Shellder</td>
      <td>30</td>
      <td>65</td>
      <td>100</td>
      <td>45</td>
      <td>25</td>
      <td>40</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>99</td>
      <td>Cloyster</td>
      <td>50</td>
      <td>95</td>
      <td>180</td>
      <td>85</td>
      <td>45</td>
      <td>70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <th>Poison</th>
      <td>100</td>
      <td>Gastly</td>
      <td>30</td>
      <td>35</td>
      <td>30</td>
      <td>100</td>
      <td>35</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 10 columns</p>
</div>



<a id="39"></a> <br>
### 데이터 프레임 회전시키기   (PIVOTING DATA FRAMES)
* pivoting: 모양 바꾸는 도구 (reshape tool)


```
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment</th>
      <th>gender</th>
      <th>response</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>F</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>M</td>
      <td>45</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>F</td>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>M</td>
      <td>9</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```
# pivoting
df.pivot(index="treatment",columns = "gender",values="response")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>gender</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>10</td>
      <td>45</td>
    </tr>
    <tr>
      <th>B</th>
      <td>5</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<a id="40"></a> <br>
### 데이터 프레임 합치기와 분할  (STACKING and UNSTACKING DATAFRAME)
* 다중 레이블 인덱스 처리
* level: 분할된 인덱스의 위치
* swaplevel: 내부 및 외부 단게 인덱스 위치 변경


```
df1 = df.set_index(["treatment", "gender"])
df1
# 이것을 분할해보자 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>response</th>
      <th>age</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th>gender</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>F</th>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>M</th>
      <td>45</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>F</th>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>M</th>
      <td>9</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```
# 단계는 인덱스(index)를 결정합니다
df.unstack(level = 0)
```




    treatment  0     A
               1     A
               2     B
               3     B
    gender     0     F
               1     M
               2     F
               3     M
    response   0    10
               1    45
               2     5
               3     9
    age        0    15
               1     4
               2    72
               3    65
    dtype: object




```
df.unstack(level=1)
```




    treatment  0     A
               1     A
               2     B
               3     B
    gender     0     F
               1     M
               2     F
               3     M
    response   0    10
               1    45
               2     5
               3     9
    age        0    15
               1     4
               2    72
               3    65
    dtype: object




```
# 외부와 내부 단계의 인덱스 위치를 바꿔보자
df2 = df1.swaplevel(0,1)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>response</th>
      <th>age</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <th>A</th>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>M</th>
      <th>A</th>
      <td>45</td>
      <td>4</td>
    </tr>
    <tr>
      <th>F</th>
      <th>B</th>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>M</th>
      <th>B</th>
      <td>9</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>



<a id="41"></a> <br>
### 데이터 프레임 녹이기(melt)  (MELTING DATA FRAMES)  *역자 : melt()함수를 사용한다는 의미같습니다.
* 중심(pivoting))의 역방향 (Reverse of pivoting)


```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment</th>
      <th>gender</th>
      <th>response</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>F</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>M</td>
      <td>45</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>F</td>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>M</td>
      <td>9</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```
# df.pivot (index ="treatment", columns = "gender", values="response")
pd.melt(df, id_vars="treatment", value_vars=["age","response"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>age</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>age</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>age</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>age</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>response</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A</td>
      <td>response</td>
      <td>45</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B</td>
      <td>response</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>response</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<a id="42"></a> <br>
### 카테고리와 그룹  (CATEGORICALS AND GROUPBY)


```
# 우리는 이걸 사용할거다.
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment</th>
      <th>gender</th>
      <th>response</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>F</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>M</td>
      <td>45</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>F</td>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>M</td>
      <td>9</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```
# "treatment"에 따라 다른 다른 열(features)들의 평균
df.groupby("treatment").mean()    # mean은 집합(aggregation))이다.  / reduction 함수.
# sum, std, max 또는 min 같은 다른 함수들도 있다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response</th>
      <th>age</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>27.5</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>B</th>
      <td>7.0</td>
      <td>68.5</td>
    </tr>
  </tbody>
</table>
</div>




```
# 우리는 오직 하나만 선택할 수도 있다.
df.groupby("treatment").age.max()
```




    treatment
    A    15
    B    72
    Name: age, dtype: int64




```
# 우리는 여러 열(features)을 선택할 수 있다
df.groupby("treatment")[["age","response"]].min()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>response</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>B</th>
      <td>65</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```
df.info()
# 보다시피 성별은 객체(object)입니다.
# 그러나 groupby를 사용하면 범주형(categorical) 데이터로 변환할 수 있습니다.
# 범주형(categorical) 데이터는 적은 메모리를 사용하므로, groupby와 같은 작업 속도를 높입니다.
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4 entries, 0 to 3
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   treatment  4 non-null      object
     1   gender     4 non-null      object
     2   response   4 non-null      int64 
     3   age        4 non-null      int64 
    dtypes: int64(2), object(2)
    memory usage: 256.0+ bytes


# 결론  (CONCLUSION)
Thank you for your votes and comments
<br> **MACHINE LEARNING ** https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners/
<br> **DEEP LEARNING** https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
<br> **STATISTICAL LEARNING** https://www.kaggle.com/kanncaa1/statistical-learning-tutorial-for-beginners
<br>**If you have any question or suggest, I will be happy to hear it.**

*역자 : 이것으로 Data_ScienceTutorial_for_Beginners 커널의 번역이 끝났습니다.
        다음 시리즈 번역에서 다시뵙겠습니다.
        감사합니다.
