# 2. 텍스트 전처리

## 2-1. 토큰화(Tokenization)

크롤링으로 얻어낸 코퍼스 데이터가 전처리 되지 않은 상태라면, 용도에 맞게 토큰화&정제*정규화를 진행

### 토큰화
- 주어진 코퍼스(corpus)에서 토큰이라 불리는 단위로 나누는 작업
- 보통 의미있는 단위로 토큰을 정의

### 단어 토큰화
- 토큰의 기준을 word로 하는 경우
  - word: 단어, 단어구, 의미를 갖는 문자열
  - 예: 구두점을 지운 후 띄어쓰기를 기준으로 잘라내기
    - 구두점: . / , / ? / ; / ! 등과 같은 기호
    - 입력: Time is illusion. Lunchtime double so!
    - 출력: "Time", "is", "an", "illusion", "Lunchtime", "double", "so"
- 구두점이나 특수문자를 전부 정제하면 토큰이 의미를 잃어버릴 수 있음
- 영어와 달리 한국어는 띄어쓰기 만으로는 단어 토큰을 구분하기 어려움

### 토큰화할 수 있는 다양한 선택지
- word_tokenize: "Don't" / "Jone's" -> 'Do', "n't" / 'Jone', "'s"
- wordPunctTokenizer: "Don't be" / "Jone's" -> 'Don', "'", 't' / 'Jone', "'", 's'
  - 구두점을 별도로 분류
- 케라스: "Don't" / "Jone's" -> "don't", "jone's"
  - 모든 알파벳을 소문자로 바꿈
  - 마침표, 컴마, 느낌표 등의 구두점 제거
  - 그러나 "'"는 보존
  
### 토큰화에서 고려해야할 사항
토큰화 작업을 단순하게 구두점을 제외하고 공백 기준으로 잘라내는 작업이라 할 수 없으며, 보다 섬세한 알고리즘이 필요
- 구두점이나 특수 문자를 단순 제외해서는 안됨
  - 구두점도 하나의 토큰으로 분류할 필요성
  - 단어 자체에 구두점을 갖고 있는 경우
  - 숫자 사이에 컴마가 들어가는 경우
- 줄임말과 단어 내에 띄어쓰기가 있는 경우
  - 그러한 단어를 하나로 인식할 수 있어야 함
  - what're = what are
  - New York
 
### 표준 토큰화 예제 - Penn Treebank Tokenization
- 규칙
  - 하이픈으로 구성된 단어는 하나로 유지
  - doesn't와 같이 "'"로 '접어'가 함께하는 단어는 분리
```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))
```
```python
트리뱅크 워드토크나이저 : ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```

### 문장 토큰화
- 토큰의 단위가 sentence인 경우
- 문장 분류(sentence segmentation)과 같은 말
- 예: Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.
  - 단순히 마침표로 문장을 구분짓기 어려움
- sent_tokenize
```python
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))
```
```python
문장 토큰화1 : ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
문장 토큰화2 : ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
> 문장 토큰화1: 성공적으로 모든 문장을 구분함<br>
> 문장 토큰화2: NLTK는 단순히 마침표를 구분자로 문장을 구분하지 않았기 때문에 PH.D.을 문장 내의 단어로 인식
- 한국어 문장 토큰화: KSS(Korean Sentence Splitter)
```python
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))
```
```python
한국어 문장 토큰화 : ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
```
> 정상적으로 모든 문장이 분리됨

### 한국어 토큰화의 어려움
- 영어는 합성어나 줄임말에 대한 예외처리만 한다면 띄어쓰기를 기준으로하는 토큰화를 수행해도 단어 토큰화가 잘 작동
- 한국어는 띄어쓰기만으로는 토큰화를 하기에 부족함
- 한국어의 띄어쓰기 단위인 '어절'에 대한 어절 토큰화는 한국어 NLP에서 지양
- 한국어는 교착어(조사, 어미 등을 붙여서 말을 만드는 언어)
- 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않음(띄어쓰기가 지켜지지 않아도 이해에 어려움이 없기 때문)

### 한국어 토큰화
**형태소(morpheme)**의 개념을 이해해야 함<br>
형태소: 뜻을 가진 가장 작은 말의 단위
- 자립 형태소: 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소
  - 그 자체로 단어가 됨
  - 체언, 수식언, 감탄사 등
- 의존 형태소: 다른 형태소와 결합하여 사용되는 형태소
  - 접사, 어미, 조사, 어간
- 단어 토큰화와 유사한 형태를 얻으려면 어절 토큰화가 아니라 형태소 토큰화를 수행해야 함

#### 품사 태깅(Part-of-speech tagging)
단어의 표기는 같지만 품사에 따라서 의미가 달라지기도 하기 때문에 해당 언어가 어떤 품사로 쓰였는지 구분해야 함, 이 작업을 품사 태깅이라 함
- KoNLPy 패키지를 통해 사용할 수 있는 형태소 분석기: Okt, Mecab, Komoran, 한나눔, 꼬꼬
- NLTK와 KoNLPy를 이용한 토큰화 실습
```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))
```
```python
단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```
```python
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
```
```python
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
```
- 관련 메소드
  - morphs: 형태소 추출
  - pos: 품사 태깅
  - nouns: 명사 추출


## 2-2 정제(Cleaning) & 정규화(Normalization)
### 정제
- 목적: 갖고 있는 코퍼스로부터 노이즈 데이터를 제거함
#### 불필요한 단어의 제거
- 노이즈 데이터
  - 자연어가 아니면서 아무 의미도 갖지 않는 글자들(특수 문자 등)
  - 분석하고자 하는 목적에 맞지 않는 불필요 단어들
- 등장 빈도가 적은 단어
  - 텍스트 데이터에서 너무 적게 등장해서 자연어 처리에 도움이 되지 않는 단어들
- 길이가 짧은 단어
  - 영어권 언어에서 길이가 짧은 단어들은 대부분 불용어에 해당
  - 길이가 짧은 단어를 제거하는 2차 이유: 길이를 조건으로 텍스트를 삭제하면서 단어가 아닌 구두점들까지도 한꺼번에 제거하기 위함
  - 영어 단어의 길이가 한국어 단어의 길이보다 평균적으로 길고, 한국어 단어는 한자어가 많아 한 글자만으로도 의미를 가진 경우가 많기에 길이가 짧은 단어를 제거하는 방법은 한국어에서는 유효하지 않을 수 있음

### 정규화
- 목적: 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어줌
#### 규칙 기반 단어 통합
- 필요에 따라 직접 코딩을 통해 정규화 규칙을 정의할 수 있음
- 예: USA == US
#### 대, 소문자 통합
- 단어의 개수를 줄일 수 있는 정규화 방법
- 주로 대문자를 소문자로 변환하는 소문자 변환작업으로 이루어짐
- 검색 엔진은 소문자 변환을 적용했을 가능성이 높기 때문에 통합이 필요
- 통합하면 안되는 경우: 대문자와 소문자가 구분되어야 하는 경우
  - 국가명, 회사명, 사람이름 등
- 결국 예외 사항을 크게 고려하지 않고 모든 코퍼스를 소문자로 바꾸는 것이 종종 더 실용적
#### 정규 표현식(Regular Expression)
- 얻어낸 코퍼스에서 노이즈 데이터의 특징을 잡아낼 수 있다면, 정규 표현식을 통해 이를 제거할 수 있는 경우가 많음


## 2-3 어간 추출(Stemming) & 표제어 추출(Lemmatization)
어간 추출과 표제어 추출은 정규화 방법<br>
의미가 다른 단어들을 하나의 단어로 일반화시킬 수 있다면 하나의 단어로 일반화시켜서 문서 내의 단어의 수를 줄이겠다는 목적을 가짐<br>
단어의 빈도수를 기반으로 문제를 풀고자 하는 자연어 처리 문제에서 주로 사용됨
### 표제어 추출
단어들이 다른 형태를 가지더라도, 그 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단<br>
표제어: 기본 사전형 단어
#### 형태학적 파싱
- 어간(stem): 단어의 의미를 담고 있는 단어의 핵심 부분
- 접사(affix): 단어에 추가적인 의미를 주는 부분
- 형태학적 파싱: 어간과 접사의 구성 요소를 분리하는 작업
- 꼭 두 가지로 분리되지 않을 수도 있음
- 어간 추출과는 달리 단어의 형태가 적절히 보존됨
- 표제어 추출기(lemmatizer)가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있음
- WordNetLemmatizer는 입력으로 단어가 동사 품사라는 사실을 알려줄 수 있음
- WordNetLemmatizer을 통한 표제어 추출 실습
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
```
```python
표제어 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
표제어 추출 후 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
```

### 어간 추출
- 어간을 추출하는 작업
- 어간 추출은 섬세한 작업이 아니기 때문에 어간 추출 후에 나오는 결과 단어는 사전에 존재하지 않을 수도 있음

 




