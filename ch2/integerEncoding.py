# from tensorflow.keras.preprocessing.text import Tokenizer

# preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(preprocessed_sentences)
# print(tokenizer.word_index)

##### One-Hot Encoding
## 단어집합(Vocabulary): 서로 다른 단어들의 집합(book,books,booked ..)
## One-Hot Encoding: 단어 집합의 크기를 Vector 표현(원하는 인덱스1, 나머지 0)

from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("나는 자연어 처리를 배운다")

word_to_index = {word:index for index,word in enumerate(tokens)}

def one_hot_encoding(word,word_to_index):
    one_hot_vector = [0]*(len(word_to_index))
    index = word_to_index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)

sub_text = "점심 먹으러 갈래 메뉴는 갈치찜 버거가 베스트야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

## 한계
# 1. 단어 개수와 비례한 vector 개수
# 2. 단어 유사도 판별 x