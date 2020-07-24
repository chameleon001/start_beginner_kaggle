# https://wikidocs.net/64517


#%%
en_text = "A Dog Run back corner near spare bedrooms"

## split 하는 3가지방법

# 1번 spacy 방법
import en_core_web_sm
nlp = en_core_web_sm.load()
import spacy
#%%
spacy_en = spacy.load('en')
# %%
def tokenize(en_text):
    return [tok.text for tok in nlp.tokenizer(en_text)]

# %%
print(tokenize(en_text))

# %%
# 2번 nltk 방법
import nltk
nltk.download('punkt')
# %%
from nltk.tokenize import word_tokenize

# %%
print(word_tokenize(en_text))

# 3번 
# %%
print(en_text.split())

#%%
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

# %%
print(kor_text.split())

# %%
