## tf(d,t): 특정문서(d)에 특정단어(t) 등장횟수
## df(t): 특정 단어(t)가 등장한 문서의 수
## idf(t): df(t)의 반비례 + 가중치에 의해서 log 취함

import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

## tf,df,idf
N = len(docs)
def tf(t,d):
    return d.count(t)
def idf(t):
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d):
  return tf(t,d)* idf(t)