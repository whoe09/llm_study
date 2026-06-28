#희소 벡터(원-핫 인코딩) -> 밀집 벡터(차원을 줄임)
#단어를 밀집벡터 = 워드 임베딩 -> 워드 임베딩 과정을 통해 나온 결과=임베딩 벡터
# 다차원 공간에 벡터화 = 분산 표현(distributed representation)

# Word2Vec 방식
## CBOW: 주변단어 입력으로 중간에 있는 단어 예측
## Skip-Gram: 중간에 있는 단어들을 입력으로 주변단어 예측


# %%
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# %%
targetXML =  open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

parse_text = '\n'.join(target_text.xpath('//content/text()'))
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)

# %%
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

result = [word_tokenize(sentence) for sentence in normalized_text]

# %%
print('총 샘플의 개수 : {}'.format(len(result)))

for line in result[:3]:
    print(line)

# Word2Vec 훈련시키기
# %%
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=result, vector_size=100,window=5, min_count=5,workers=4,sg=0)
# vector_size: 임베딩 된 벡터의 차원(M)
# window: 컨텍스트 윈도우 크기
# min_count: 단어 최소 빈도수(n이하는 학습x)
# workers: 학습을 위한 프로세스 수
# sg: 0=CBOW, 1=Skip-gram

# wv.most_simiar(""): 유사한 단어 리스트
model_result = model.wv.most_similar("boy")
print(model_result)

# %%
model.wv.save_word2vec_format('eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드


# 한국어 Word2Vec
# %%
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

# %%
train_data = pd.read_table('ratings.txt')
print(train_data[:5])

print(len(train_data))

# %%
print(train_data.isnull().values.any())
train_data=train_data.dropna(how='any')
print(train_data.isnull().values.any())
#%%
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
train_data[:5]

#%%
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
from tqdm import tqdm

okt = Okt()
tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence,stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    tokenized_data.append(stopwords_removed_sentence)
#%%
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers','footers','quotes'))
documents = dataset.data
print('총 샘플 수 :',len(documents))

#%%
nltk.download('stopwords')
news_df = pd.DataFrame({'document':documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ", regex=True)
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

news_df.replace("", float("NaN"), inplace=True)
news_df.isnull().values.any()

news_df.dropna(inplace=True)
print('총 샘플 수 :',len(news_df))
#%%

# 불용어를 제거
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()
#%%
# numpy 구버전: 리스트의 길이가 달라도 object로 처리해줬음
# drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
# tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)
# print('총 샘플 수 :',len(tokenized_doc))

tokenized_doc = [sentence for sentence in tokenized_doc if len(sentence) > 1]
print('총 샘플 수 :', len(tokenized_doc))

#%%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)

#%%
from tensorflow.keras.preprocessing.sequence import skipgrams
# 네거티브 샘플링
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0], 
          idx2word[pairs[i][1]], pairs[i][1], 
          labels[i]))
    
# %%
## 워드 임베딩
vocab_size = 20000
output_dim = 128
input_length = 500

v = Embedding(vocab_size, output_dim, input_length=input_length)
# vocab_size: 텍스트 데이터의 전체 단어 집합 크기
# output_dim: 워드 임베딩 후 벡터 차원
# input_length: 입력 시퀀스 길이(샘플 길이)

# %%
# 긍정문 = 1, 부정문 = 0
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

vocab_size = len(tokenizer.word_index) + 1 # 패딩을 고려하여 +1
print('단어 집합 :',vocab_size)
# %%
X_encoded = tokenizer.texts_to_sequences(sentences)
print('정수 인코딩 결과: ',X_encoded)
# %%
# 패딩 진행
max_len = max(len(l) for l in X_encoded)
X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)
print('패딩 결과 :')
print(X_train)

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

embedding_dim = max_len

modle = Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)