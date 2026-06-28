# from konlpy.tag import Okt
# from konlpy.tag import Kkma

# okt = Okt()
# kkma = Kkma()

# print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

# morpsh: 형태소 추출
# pos: 품사 태킹 (value,what)
# nonus: 명사 

################## Cleaning
import re
text = "I was wondering if anyone out there could enlighten me on this car."

# 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('',text))

################# Stremming / Lemmatization
# Lemmatization 표제어
## 기본 사전형 단어, 뿌리가 되는 단어이다.

#stemming 어간추출
## 어림짐작 버전 추출

###### Integer Encoding
