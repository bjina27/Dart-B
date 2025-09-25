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
- 포터 알고리즘의 어간추출이 가지는 규칙
  - ALIZE -> AL / formalize -> formal
  - ANCE -> 제거 / allowance -> allow
  - ICAL -> IC / eletricical -> electric
- 포터 알고리즘
  - 정밀하게 설계되어 정확도가 높음, 영어 자연어 처리에서는 가장 준수한 선택
- PorterStemmer과 LancasterStemmer 비교
```python
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('어간 추출 전 :', words)
print('포터 스테머의 어간 추출 후:',[porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머의 어간 추출 후:',[lancaster_stemmer.stem(w) for w in words])
```
```python
어간 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
포터 스테머의 어간 추출 후: ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
랭커스터 스테머의 어간 추출 후: ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
```
#### 한국어에서의 어간 추출
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/8fb4295a-cdee-4abb-b776-b32320fe6bec" />

- 5언 9품사
- 용언은 어간과 어미의 결합으로 구성됨
- 활용(conjugation)
  - 어간이 어미를 가지는 일
  - 어간이 어미를 취할 때, 어간의 모습이 일정하다면 규칙 활용, 어간이나 어미의 모습이 변하면 불규칙 활용으로 나뉨 


## 2-4 불용어(Stopword)
갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요(I, my, me, over, 조사, 접미사 등)<br>
불용어는 개발자가 직접 정의 가능
### NLTK에서 불용어 확인하기
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt
stop_words_list = stopwords.words('english') // NLTK가 정의한 불용어 리스트 리턴
print('불용어 개수 :', len(stop_words_list))
print('불용어 10개 출력 :',stop_words_list[:10])
```
```python
불용어 개수 : 179
불용어 10개 출력 : ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
```
### NLTK를 통해서 불용어 제거하기
```python
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example) ## 단어 토큰화

result = []
for word in word_tokens: 
    if word not in stop_words: 
        result.append(word) 

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)
```
```python
불용어 제거 전 : ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
불용어 제거 후 : ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```
### 한국어에서 불용어 제거하기
직접 불용어를 정의하고, 주어진 문장으로부터 사용자가 정의한 불용어 사진으로부터 불용어를 제거
```python
okt = Okt()

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))
word_tokens = okt.morphs(example)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)
```
```python
불용어 제거 전 : ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한', '게', '있지', '.']
불용어 제거 후 : ['고기', '하면', '.', '고기', '라고', '다', '아니거든', '.', '예컨대', '삼겹살', '을', '중요한', '있지', '.']
```


## 2-5 정규 표현식(Regular Expression)
### 정규 표현식 문법과 모듈 함수
파이썬에서는 정규 표현식 모듈 re를 지원하므로, 이를 이용하면 특정 규칙이 있는 텍스트 데이터를 빠르게 정제할 수 있음

### 정규 표현식 문법
~~~
<특수문자 설명>
.  한 개의 임의의 문자를 나타냅니다. (줄바꿈 문자인 \n는 제외)
?  앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있습니다. (문자가 0개 또는 1개)
*  앞의 문자가 무한개로 존재할 수도 있고, 존재하지 않을 수도 있습니다. (문자가 0개 이상)
+  앞의 문자가 최소 한 개 이상 존재합니다. (문자가 1개 이상)
^  뒤의 문자열로 문자열이 시작됩니다.
$  앞의 문자열로 문자열이 끝납니다.
{숫자}  숫자만큼 반복합니다.
{숫자1, 숫자2}  숫자1 이상 숫자2 이하만큼 반복합니다. ?, *, +를 이것으로 대체할 수 있습니다.
{숫자,}  숫자 이상만큼 반복합니다.
[ ]  대괄호 안의 문자들 중 한 개의 문자와 매치합니다. [amk]라고 한다면 a 또는 m 또는 k 중 하나라도 존재하면 매치를 의미합니다. [a-z]와 같이 범위를 지정할 수도 있습니다. [a-zA-Z]는 알파벳 전체를 의미하는 범위이며, 문자열에 알파벳이 존재하면 매치를 의미합니다.
[^문자]  해당 문자를 제외한 문자를 매치합니다.
l  AlB와 같이 쓰이며 A 또는 B의 의미를 가집니다.
<문자규칙 설명>
\\\  역 슬래쉬 문자 자체를 의미합니다
\\d  모든 숫자를 의미합니다. [0-9]와 의미가 동일합니다.
\\D  숫자를 제외한 모든 문자를 의미합니다. [^0-9]와 의미가 동일합니다.
\\s  공백을 의미합니다. [ \t\n\r\f\v]와 의미가 동일합니다.
\\S  공백을 제외한 문자를 의미합니다. [^ \t\n\r\f\v]와 의미가 동일합니다.
\\w  문자 또는 숫자를 의미합니다. [a-zA-Z0-9]와 의미가 동일합니다.
\\W  문자 또는 숫자가 아닌 문자를 의미합니다. [^a-zA-Z0-9]와 의미가 동일합니다.
<모듈함수 설명>
re.compile()	정규표현식을 컴파일하는 함수입니다. 다시 말해, 파이썬에게 전해주는 역할을 합니다. 찾고자 하는 패턴이 빈번한 경우에는 미리 컴파일해놓고 사용하면 속도와 편의성면에서 유리합니다.
re.search()	문자열 전체에 대해서 정규표현식과 매치되는지를 검색합니다.
re.match()	문자열의 처음이 정규표현식과 매치되는지를 검색합니다.
re.split()	정규 표현식을 기준으로 문자열을 분리하여 리스트로 리턴합니다.
re.findall()	문자열에서 정규 표현식과 매치되는 모든 경우의 문자열을 찾아서 리스트로 리턴합니다. 만약, 매치되는 문자열이 없다면 빈 리스트가 리턴됩니다.
re.finditer()  문자열에서 정규 표현식과 매치되는 모든 경우의 문자열에 대한 이터레이터 객체를 리턴합니다.
re.sub()	문자열에서 정규 표현식과 일치하는 부분에 대해서 다른 문자열로 대체합니다.
~~~

### 정규 표현식 텍스트 전처리 예제
```python
text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""
re.split('\s+', text)  
re.findall('\d+',text)  
re.findall('[A-Z]',text)
re.findall('[A-Z]{4}',text)  
re.findall('[A-Z][a-z]+',text)
```
```python
['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']
['100', '101', '102]
['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']
['PROF', 'STUD', 'STUD']
['John', 'James', 'Mac'] 
```

### 정규 표현식을 이용한 토큰화
- NLTK에서는 정규 표현식을 사용해서 단어 토큰화를 수행하는 RegexpTokenizer을 지원함
- RegexpTokenizer()에서 괄호 안에 하나의 토큰으로 규정하기를 원하는 정규 표현식을 넣어서 토큰화를 수행
```python
from nltk.tokenize import RegexpTokenizer

text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"

tokenizer1 = RegexpTokenizer("[\w]+")
tokenizer2 = RegexpTokenizer("\s+", gaps=True)

print(tokenizer1.tokenize(text))
print(tokenizer2.tokenize(text))
```
```python
['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```


## 2-6 정수 인코딩(Integer Encoding)
### 정수 인코딩
- 단어에 정수를 부여하는 방법 중 하나
- 단어를 빈도수 순으로 정렬한 단어 집합(vocabulary)를 만들고, 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여
#### dictionary 사용하기
```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
sentences = sent_tokenize(raw_text)
print(sentences)

# 전처리
vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    preprocessed_sentences.append(result) 
print(preprocessed_sentences)

# vocab(각 단어에 대한 빈도수) 출력
print('단어 집합 :',vocab)

# 'barber'라는 단어의 빈도수 출력
print(vocab["barber"])

# 빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)

# 높은빈도수를 가진 단어일 수록 낮은 정수를 부여, 정수는 1부터 부여
word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 빈도수가 작은 단어는 제외.
        i = i + 1
        word_to_index[word] = i

print(word_to_index) ## 1의 인덱스를 가진 단어가 가장 빈도수가 높은 단

# 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우
vocab_size = 5

# 인덱스가 5 초과인 단어 제거
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

# 해당 단어에 대한 인덱스 정보를 삭제
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)

# 단어 집합에 존대하지 않는 단어들이 생기는 상황(Out-Of-Vocabulary, OOV 문제)
# word_to_index에 'OOV'라는 단어를 새롭게 추가, 단어 집합에 없는 단어들을 'OOV'의 인덱스로 인코딩
word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)

# word_to_index를 사용하여 sentences의 모든 단어들을 맵핑되는 정수로 인코딩
encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)
```

#### Counter 사용하기
```python
from collections import Counter
print(preprocessed_sentences)
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# vocabulary를 만들기 위해 sentences에서 문장의 경계인 [,]을 제거하고 단어들을 하나의 리스트로 만들기
# words = np.hstack(preprocessed_sentences)으로도 수행 가능.
all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)

# 파이썬의 Counter 모듈을 이용하여 단어의 빈도수 카운트
vocab = Counter(all_words_list)
print(vocab)

# print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력
print(vocab["barber"])

# 등장 빈도수 상위 5개 단어만 단어 집합으로 저장
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
vocab

# 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스 부여
word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i

print(word_to_index)
```

#### NLTK의 FreqDist 사용하기
```python
from nltk import FreqDist
import numpy as np
# np.hstack으로 문장 구분을 제거
vocab = FreqDist(np.hstack(preprocessed_sentences))

print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
print(vocab)

# enumerate()를 사용하여 좀 더 짧은 코드로 인덱스 부여
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)
```

#### enumerate 이해하기
enumerate()는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스를 순차적으로 함께 리턴
```python
test_input = ['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test_input): # 입력의 순서대로 0부터 인덱스를 부여함.
  print("value : {}, index: {}".format(value, index))
```
```python
value : a, index: 0
value : b, index: 1
value : c, index: 2
value : d, index: 3
value : e, index: 4
```

### 케라스(Keras)의 텍스트 전처리
때로는 정수 인코딩을 위해 케라스의 전처리 도구인 토크나이저를 사용하기도 함
```python
from tensorflow.keras.preprocessing.text import Tokenizer
preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()

# fit_on_texts()안에 코퍼스를 입력으로 하면 단어 빈도수가 높은 순으로 낮은 정수 인덱스 부여
tokenizer.fit_on_texts(preprocessed_sentences) 

# 각 단어에 인덱스가 어떻게 부여되었는지 확인
print(tokenizer.word_index)

# 각 단어가 카운트룰 수행하였을 때 몇 개였는지 확인
print(tokenizer.word_counts)

# 입력으로 들어온 코퍼스에 대해 각 단어를 이미 정해진 인덱스로 변
print(tokenizer.texts_to_sequences(preprocessed_sentences))

# 상위 5개 단어를 사용한다고 토크나이저를 재정의
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences)
# num_words에서 +1을 더해서 값을 넣어주는 이유는 num_words는 숫자를 0부터 카운트합니다.
# 만약 5를 넣으면 0 ~ 4번 단어 보존을 의미하게 되므로 뒤의 실습에서 1번 단어부터 4번 단어만 남게됩니다.
# 그렇기 때문에 1 ~ 5번 단어까지 사용하고 싶다면 num_words에 숫자 5를 넣어주는 것이 아니라 5+1인 값을 넣어주어야 합니다.

# 코퍼스에 대해 정수 인코딩 진행
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```


## 2-7 패딩(Padding)
자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있음<br>
기계는 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고, 한꺼번에 묶으서 처리할 수 있음<br>
따라서 병렬 연산을 위해 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요
### Numpy로 패딩하기
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# 단어 집합을 만들고, 정수 인코딩 수행
ed!tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

# 모두 동일한 길이로 맞춰주기 위해 이 중에서 가장 길이가 긴 문장의 길이를 계산
max_len = max(len(item) for item in encoded)
print('최대 길이 :',max_len)

# 가장 길이가 긴 문장의 길이보다 짧은 문장에는 숫자 0을 채워서 길이 7로 맞춰줌
for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np
```
- 0번 단어는 아무런 의미가 없는 단어이기 때문에 기계는 0번 단어를 무시
- 이와 같이 데이터에 특정 값을 채워서 데이터의 크기를 조정하는 것을 패딩이라고 함
- 숫자 0을 사용한다면 제로 패딩(zero padding)이라 함

### 케라스 전처리 도구로 패딩하기
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

# 케라스의 pad_sequences를 사용하여 패딩
padded = pad_sequences(encoded)
padded

# pad_sequences는 기본적으로 문서의 뒤에 0을 채우는 것이 아니라 앞에 0으로 채움
# 뒤에 0을 채우고 싶다면 인자로 padding='post'를 주면 
padded = pad_sequences(encoded, padding='post')
padded

# 길이에 제한을 두고 패딩하는 경우: maxlen의 인자로 정수를 주면 됨
padded = pad_sequences(encoded, padding='post', maxlen=5)
padded

# 데이터가 손실될 경우 앞의 단어가 아니라 뒤의 단어가 삭제되도록 하고 싶다면 truncating이라는 인자를 사용
# truncating='post'를 사용할 경우 뒤의 단어가 삭제됨
padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
padded

# 숫자 0이 아니라 다른 숫자를 사용하는 것이 가능
# 현재 사용된 정수들과 겹치지 않도록 단어 집합의 크기에 +1을 한 숫자로 사용
last_value = len(tokenizer.word_index) + 1 # 단어 집합의 크기보다 1 큰 숫자를 사용
print(last_value)

# pad_sequences의 인자로 value를 사용하면 0이 아닌 다른 숫자로 패딩 가능
padded = pad_sequences(encoded, padding='post', value=last_value)
padded
```


## 2-8 원-핫 인코딩(One-Hot Encoding)
### 단어 집합(vocabulary)
- 서로 다른 단어들의 집합
- book != book

### 원-핫 인코딩이란?
- 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스의 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
- 원-핫 인코딩을 통해 표현된 벡터를 원-핫 벡터라고 함
- 원-핫 인코딩의 두 가지 과정
  - 정수 인코딩을 수행(각 단어에 고유한 정수를 부여)
  - 표현하고 싶은 단어의 고유한 정수를 인덱스로 간주하고 해당 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여
```python
from konlpy.tag import Okt  

okt = Okt()  
tokens = okt.morphs("나는 자연어 처리를 배운다")  
print(tokens)

# 각 토큰에 대해 고유한 정수를 부여
word_to_index = {word : index for index, word in enumerate(tokens)}
print('단어 집합 :',word_to_index)

# 토큰을 입력하면 해당 토큰에 대한 원-핫 벡터를 만들어내는 함수를 만듦
def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

# '자연어'라는 단어의 원-핫 벡터 얻기
one_hot_encoding("자연어", word_to_index)
```
```python
[0, 0, 1, 0, 0, 0]  
```

### 케라스를 이용한 원-핫 인코딩
케라스는 원-핫 인코딩을 수행하는 유용한 도구 to_categorical()를 지원함
```python
text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

# 케라스 토크나이저를 이용한 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)

# 생성된 단어 집합 내의 일부 단어들로만 구성된 서브 텍스트를 만들어 확인
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

# 케라스의 to_categorical()을 사용하여 원-핫 인코딩
one_hot = to_categorical(encoded)
print(one_hot)
```
```python
[[0. 0. 1. 0. 0. 0. 0. 0.] # 인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] # 인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] # 인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] # 인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] # 인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] # 인덱스 7의 원-핫 벡터
```

### 원-핫 인코딩의 한계
- 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어남, 비효율적
- 단어의 유사도를 표현하지 못함
  - 검색 시스템 등에서 문제가 될 소지가 있음: '삿포로 숙소'를 입력하면 '삿포로 호텔'과 같은 유사 단어에 대한 결과를 보여주지 못함
 

## 2-9 데이터 분리(Splitting Data)
### 지도 학습(Supervised Learning)
지도 학습의 훈련 데이터는 정답이 무엇인지 맞춰야 하는 '문제'에 해당되는 데이터와 레이블이라고 부르는 '정답'이 적혀있는 데이터로 구성되어 있음
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/4166a478-5cea-4786-9b71-40ea7558c51f" />

- 데이터 분리의 과정
  - 메일의 내용이 담긴 열을 X에, 메일의 스팸 여부(정답)가 적힌 열을 Y에 저장
  - X와 Y에 대해 일부 데이터를 test 데이터로 분리
  - 분리 시에는 여전히 X와 Y의 맵핑 관계를 유지해야 함
  - train의 18000개의 X, Y의 Tkdrhk test의 2000개의 X, Y 쌍이 생성됨
    - X_train: 문제지 데이터
    - Y_train: 문제지에 대한 정답 데이터
    - X_test: 시험지 데이터
    - Y_test: 시험지에 대한 정답 데이터
  - 기계는 이제부터 X_train과 Y_train에 대해 학습
  - 학습을 다 한 기계에게 Y_test는 보여주지 않고 X_test에 대해서 정답을 예측하게 함
  - 기계가 예측한 답과 실제 정답인 Y_test를 비교하며 기계가 얼마나 맞췄는지 평가
  - 평가한 수치는 기계의 정확도(Accuracy)가 됨

 ### X와 Y 분리하기
 #### zip 함수를 이용하여 분리
 ```python
# zip(): 동일한 개수를 가지는 시퀀스 자료형에서 각 순서에 등장하는 원소들끼리 묶어주는 역할
X, y = zip(['a', 1], ['b', 2], ['c', 3])
print('X 데이터 :',X)
print('y 데이터 :',y)

# 각 데이터에서 첫번째로 등장한 원소들끼리 묶이고, 두번째로 등장한 원소들끼리 묶임
sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences)
print('X 데이터 :',X)
print('y 데이터 :',y)
```

#### 데이터 프레임을 이용하여 분리
```python
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df

X = df['메일 본문']
y = df['스팸 메일 유무']

```
#### Numpy를 이용하여 분리
```python
# 임의의 데이터를 만들어서 Numpy의 슬라이싱(slicing)을 사용하여 데이터를 분리
np_array = np.arange(0,16).reshape((4,4))
print('전체 데이터 :')
print(np_array)

# 마지막 열을 제외하고 X데이터에 저장, 마지막 열만을 Y 데이터에 저장
X = np_array[:, :3]
y = np_array[:,3]

print('X 데이터 :')
print(X)
print('y 데이터 :',y)
```

### 테스트 데이터 분리하기
#### 사이킷 런을 이용하여 분리하기
~~~
X : 독립 변수 데이터. (배열이나 데이터프레임)
y : 종속 변수 데이터. 레이블 데이터.
test_size : 테스트용 데이터 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
train_size : 학습용 데이터의 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
random_state : 난수 시드
~~~
```python
# 임의로 X와 y 데이터를 생성
X, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))

# 7:3의 비율로 훈련 데이터와 테스트 데이터 분리
# train_test_split()은 기본적으로 데이터의 순서를 섞고나서 훈련 데이터와 테스트 데이터를 분리
# 만약, random_state의 값을 특정 숫자로 기재해준 뒤 다음에도 동일한 숫자로 기재하면 항상 동일한 훈련 데이터와 테스트 데이터를 얻을 수 있음
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
```
#### 수동으로 분리하기
```python
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
X, y = np.arange(0,24).reshape((12,2)), range(12)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))

# 훈련 데이터의 개수와 테스트 데이터의 개수 정하기
num_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
num_of_test = int(len(X) - num_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print('훈련 데이터의 크기 :',num_of_train)
print('테스트 데이터의 크기 :',num_of_test)

X_test = X[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
y_test = y[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
y_train = y[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
```


## 2-10 한국어 전처리 패키지(Text Preprocessing Tools for Korean Text)
### PyKoSpacing
- 띄어쓰기가 되어있지 않은 문장을 띄어쓰기를 한 문장으로 변환해주는 패키지
- 대용량 코퍼스를 학습하여 만들어진 띄어쓰기 딥러닝 모델로, 준수한 성능을 가짐
### SOYNLP를 이용한 단어 토큰화
품사 태깅, 단어 토큰화 등을 지원<br>
비지도 학습으로 단어 토큰화<br>
데이터에 자주 등장하는 단어들을 단어로 분석<br>
soynlp 단어 토크나이저는 내부적으로 단어 점수 표로 동작하고, 이 점수는 응집 확률과 브랜칭 엔트로피를 활용
#### 기존의 형태소 분석기가 가진 문제
신조어 문제: 형태소 분석기에 등록되지 않은 단어는 제대로 구분하지 못함
#### 학습하기
```python
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

# 훈련 데이터를 다수의 문서로 분리
corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)

# 상위 3개의 문서만 출력
i = 0
for document in corpus:
  if len(document) > 0:
    print(document)
    i = i+1
  if i == 3:
    break

# WordExtractor.extract()를 통해서 전체 코퍼스에 대해 단어 점수표를 계산
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
```
#### SOYNLP의 응집 확률(cohesion probability)
- 응집 확률
  - 내부 문자열이 얼마나 응집하여 자주 등장하는지를 판단하는 척도
  - 문자열을 문자 단위로 분리하여 내부 문자열을 만드는 과정에서 왼쪽부터 순서대로 문자를 추가하며 각 문자열이 주어졌을 때 그 다음 문자가 나올 확률을 계산하여 누적곱을 한 값
  - 이 값이 높을수록 전체 코퍼스에서 이 문자열 시퀀스는 하나의 단어로 등장할 가능성이 높음
```python
# '반포한'의 응집확률 계산
word_score_table["반포한"].cohesion_forward

# '반포한강'의 응집확률 계산
word_score_table["반포한강"].cohesion_forward
```
#### SOYNLP의 브랜칭 엔트로피(branching entropy)
- 확률 분포의 엔트로피 값을 사용
- 주어진 문자열에서 얼마나 다음 문자가 등장할 수 있는지를 판단하는 척도
- 브랜칭 엔트로피의 값은 하나의 완성된 단어에 가까워질수록 문맥으로 인해 점점 정확히 예측할 수 있게 되면서 점점 줄어드는 양상을 보임

#### SOYNLP의 L tokenizer
- 한국어는 띄어쓰기 단위로 나눈 어절 토큰은 주로 L 토큰 + R 토큰의 형식을 가질 때가 많음
- L 토크나이저는 L 토큰 + R 토큰으로 나누되, 분리 기준을 점수가 가장 높은 L 토큰을 찾아내는 원리임

#### 최대 점수 토크나이저
- 띄어쓰기가 되지 않는 문장에서 점수가 높은 글자 시퀀스를 순차적으로 찾아내는 토크나이저

### SOYNLP를 이용한 반복되는 문자 정제
- 문제: 'ㅋㅋㅋ', 'ㅋㅋ ㅋㅋ ㅋㅋ ㅋ'와 같은 경우를 모두 서로 단어로 처리하는 것은 불필요
- 반복되는 것은 하나로 정규

### Customized KoNLPy
- 사용자 사전을 추가하는 방법은 형태소 분석기마다 다름
- 복잡한 경우가 많음
- Customized KoNLPy: 사용자 추가가 매우 쉬운 패키지



# 7

# 8. 순환 신경망(Recurrent Neural Network)

## 8-1 순환 신경망(Recurrent Neural Network, RNN)
### RNN
입력과 출력을 시퀀스(Sequence) 단위로 처리하는 시퀀스 모델<br>
시퀀스: 번역기의 입력은 번역하고자 하는 단어의 시퀀스인 문장임, 출력에 해당되는 번역된 문장 또한 단어의 시퀀스임<br>
RNN은 가장 기본적인 인공 신경망 시퀀스 모델
- 피드 포워드 신경망(Feed Forward Neural Network): 은닉층에서 활성화 함수를 지닌 값은 오직 출력층 방향으로만 향하는 신경망, RNN은 그렇지 않은 신경망 중 하나
- RNN: 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보냄
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/a50564cc-e034-4e7b-9d25-90459a593beb" />

- x: 입력층의 입력 벡터 / y: 출력층의 출력 벡터
- 실제로는 편향 b도 입력으로 존재할 수 있음
#### cell
- RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드
- 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행
- **메모리 셀** 또는 **RNN 셀**이라고 표현
- 각각의 시점(time step)에서 바로 이전 시점에서의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 함
- 이는 현재 시점 t에서의 메모리 셀이 갖고있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임
#### 은닉 상태(hidden state)
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/f7c96054-362f-4e2e-8db8-3cde52d8d8b8" />

- 메모리 셀이 출력층 방향 또는 다음 시점인 t+1의 자신에게 보내는 값
- t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태값을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용
- 피드 포워드 신경망에서는 뉴런이라는 단위를 사용했지만, RNN에서는 뉴런 대신 입력층과 출력층에서는 각각 입력 벡터와 출력 벡터, 은닉층에서는 은닉 상태라는 표현 사용
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/8f205b69-57d6-4592-91f9-41f23b6c4189" />

- 입력 벡터의 차원이 4, 은닉 상태의 크기가 2, 출력층의 출력 벡터의 차원이 2인 RNN이 시점이 2일때의 모습
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/97164c40-8f89-4f57-9998-9b053cd18281" />

- RNN은 입력과 출력의 길이를 다르게 설계할 수 있음
- RNN 셀의 각 시점의 입, 출력 단위는 사용자가 정의하기 나름이지만, 가장 보편적인 단위는 '단어 벡터'
- 일대다 구조의 모델
  - 하나의 이미지 입력에 대해서 사진의 제목을 출력하는 이미지 캡셔닝 작업에 사용 가능
- 다대일 구조의 모델
  - 입력 문서가 긍정적인지 부정적인지를 판별하는 감성 분류, 메일이 정상 메일인지 스팸 메일인지 판별하는 스팸 메일 분류 등에 사용
- 다대다 구조의 모델
  - 사용자가 문장을 입력하면 대답 문장을 출력하는 챗봇과 입력 문장으로부터 번역된 문장을 출력하는 번역기, 태깅 작업 챕터의 개체명 인식이나 품사 태깅과 같은 작업이 속함
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/b8a27328-89ea-4c2a-a226-bce7c6a1e4fb" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/f68291a7-f76c-4e16-a4bb-875386b2af2d" />

  - ht: 현재 시점 t에서의 은닉 상태값
  - 은닉층의 메모리 셀은 ht를 계산하기 위해서 총 두개의 가중치를 가짐
    - Wx: 입력층을 위한 가중치
    - Wh: 이전 시점 t-1의 은닉 상태값인 ht-1을 위한 가중치
- RNN의 은닉층 연산(벡터와 행렬)
  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/bf7a7771-bd84-48f9-a430-8a3afdf86da8" />
  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/ed0bbad5-207d-4a53-96af-9adbb307a368" />

  - 상단 이미지: 각 벡터와 행렬의 크기
  - 하단 이미지: 배치 크기가 1이고, d와 Dh 두 값 모두를 4라고 가정하였을 때, RNN의 은닉층 연산
  - Wx, Wh, Wy의 값은 하나의 층에서는 모든 시점에서 값을 동일하게 공유
  - 은닉층이 2개 이상일 경우, 각 은닉층에서의 가중치는 서로 다름
  - 출력층
    - 이진 분류를 해야하는 경우라면 출력층에 로지스틱 회귀를 사용하여 시그모이드 함수 사용
    - 다중 클래스 분류를 해야하는 경우라면 출력층에 소프트맥스 회귀를 사용하여 소프트맥스 함수 사용

### 케라스(Keras)로 RNN 구현하기
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/5cd97374-1126-4672-9967-b1cc49c5e4cd" />

```python
# 추가 인자를 사용할 때
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))

# 다른 표기
model.add(SimpleRNN(hidden_units, input_length=M, input_dim=N))
```
~~~
hidden_units = 은닉 상태의 크기를 정의. 메모리 셀이 다음 시점의 메모리 셀과 출력층으로 보내는 값의 크기(output_dim)와도 동일. RNN의 용량(capacity)을 늘린다고 보면 되며, 중소형 모델의 경우 보통 128, 256, 512, 1024 등의 값을 가짐
timesteps = 입력 시퀀스의 길이(input_length)라고 표현하기도 함, 시점의 수
input_dim = 입력의 크기
~~~
- RNN 층은 (batch_size, timesteps, input_dim) 크기의 3D 텐서를 입력으로 받음
  - batch_size: 한 번에 학습하는 데이터의 개수
- 주의할 점
  - 위 코드는 출력층까지 포함한 인공신경망 코드가 아니라 주로 은닉층으로 간주할 수 있는 하나의 RNN 층에 대한 코드임
  - 해당 코드가 리턴하는 결과값은 하나의 은닉 상태 또는 정의하기에 따라 여러 개의 시점의 은닉 상태임
- RNN 층이 입력 3D 텐서를 입력받아서 은닉 상태를 출력하는 방식
  - 메모리 셀의 최종 시점의 은닉 상태만을 리턴하고자 한다면 (batch_size, output_dim) 크기의 2D 텐서를 리턴
  - 메모리 셀의 각 시점(time step)의 은닉 상태값들을 모아서 전체 시퀀스를 리턴하고자 한다면 (batch_size, timesteps, output_dim) 크기의 3D 텐서를 리턴
- 마지막 은닉 상태만 전달하도록 하면 다대일 문제를 풀 수 있으며, 모든 시점의 은닉 상태를 전달하도록 하면 다음층에 RNN 은닉층이 하나 더 있는 경우이거나 다대다 문제를 풀 수 있음

### 파이썬으로 RNN 구현하기
Numpy로 RNN층 구현
<img width="400" height="100" alt="image" src="https://github.com/user-attachments/assets/2d2b368c-eb20-438f-bc1b-1dd47556fd3d" />

```python
# 아래의 코드는 가상의 코드(pseudocode)로 실제 동작하는 코드가 아님. 

hidden_state_t = 0 # 초기 은닉 상태를 0(벡터)로 초기화
for input_t in input_length: # 각 시점마다 입력을 받는다.
    output_t = tanh(input_t, hidden_state_t) # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
    hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉 상태가 된다.

# 초기 은닉 상태를 출력
import numpy as np

timesteps = 10
input_dim = 4
hidden_units = 8

# 입력에 해당되는 2D 텐서
inputs = np.random.random((timesteps, input_dim))

# 초기 은닉 상태는 0(벡터)로 초기화
hidden_state_t = np.zeros((hidden_units,)) 

print('초기 은닉 상태 :',hidden_state_t)

# 가중치와 편향을 각 크기에 맞게 정의하고 크기를 출력
Wx = np.random.random((hidden_units, input_dim))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_units, hidden_units)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_units,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).

print('가중치 Wx의 크기(shape) :',np.shape(Wx))
print('가중치 Wh의 크기(shape) :',np.shape(Wh))
print('편향의 크기(shape) :',np.shape(b))

# 모든 시점의 은닉 상태를 출력한다고 가정하고 RNN층 동작
total_hidden_states = []

# 각 시점 별 입력값.
for input_t in inputs:

  # Wx * Xt + Wh * Ht-1 + b(bias)
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b)

  # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep t, output_dim)
  # 각 시점의 은닉 상태의 값을 계속해서 누적
  total_hidden_states.append(list(output_t))
  hidden_state_t = output_t

# 출력 시 값을 깔끔하게 해주는 용도.
total_hidden_states = np.stack(total_hidden_states, axis = 0) 

# (timesteps, output_dim)
print('모든 시점의 은닉 상태 :')
print(total_hidden_states)
```

### 깊은 순환 신경망(Deep Recurrent Neural Network)
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/325097ba-c717-40d0-ae85-cec1297ac626" />

순환 신경망에서 은닉층이 1개 더 추가되어 은닉층이 2개인 깊은 순환 신경망의 이미지
```python
# 은닉층을 2개 추가하는 코드
# 첫번째 은닉층은 다음 은닉층이 존재하므로 return_sequences = True를 설정하여 모든 시점에 대해서 은닉 상태 값을 다음 은닉층으로 보내줌
model = Sequential()
model.add(SimpleRNN(hidden_units, input_length=10, input_dim=5, return_sequences=True))
model.add(SimpleRNN(hidden_units, return_sequences=True))
```

### 방향 순환 신경망(Bidirectional Recurrent Neural Network)
시점 t에서의 출력값을 예측할 때 이전 시점의 입력뿐만 아니라, 이후 시점의 입력 또한 예측에 기여할 수 있다는 아이디어에 기반
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/59351bda-8e24-4a8c-96de-d67b60b7f6c7" />

- 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용
- 첫번째 메몰 셀: 앞 시점의 은닉 상태를 전달받아 현재의 은닉 상태를 계산
- 두번째 메모리 셀: 뒤 시점의 은닉 상태를 전달 받아 현재의 은닉 상태를 계산(입력 시퀀스를 반대 방향으로 읽는 것)
- 양방향 RNN도 다수의 은닉층을 가질 수 있음
```pythonLS
# 은닉층이 4개인 경우
model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True), input_shape=(timesteps, input_dim)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
```
 

## 8-2 장단기 메모리(Long Short-Term Memory, LSTM)
RNN: 앞서 배운 가장 단순한 형태의 RNN, 바닐라 RNN<br>
LSTM: 바닐라 RNN의 한계를 극복하기 위한 RNN의 변형

### 바닐라 RNN의 한계
- 출력 결과가 이전의 계산 결과에 의존
- 비교적 짧은 시퀀스에 대해서만 효과를 보임
- 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못함
- RNN이 충분한 기억력을 가지고 있지 못하다면 다음 단어를 엉뚱하게 예측하는데, 이를 장기 의존성 문제(the problem of Long-Term Dependencies)라 함

### 바닐라 RNN 내부 열어보기
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/88bb7d02-0c32-4bcd-b6b5-c2399fc26845" />

- 바닐라 RNN은 xt와 ht-1이라는 두개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 됨
- 이를 하이퍼볼릭탄젠트 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 은닉 상태가 됨

### LSTM(Long Short-Term Memory)
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/00052e36-9302-45c4-b941-19b1ee3f25ff" />

- 전통적인 RNN의 단점을 보완한 RNN의 일종
- 은닉층에 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야 할 것들을 정함
- 은닉 상태를 계산하는 식이 RNN보다 조금 더 복잡함
- 셀 상태(cell state)라는 값을 추가함
- Ct: t 시점의 셀 상태
- RNN에 비해 긴 시퀀스의 입력을 처리하는데 탁월함
- 이전시점의 셀 상태가 다음 시점의 셀 상태를 구하기 위한 입력으로 사용됨
- 은닉 상태의 값과 셀 상태의 값을 구하기 위해 새로 추가 된 3개의 게이트를 사용
  - 공통적으로 시그모이드 함수가 존재
  - 삭제 게이트
  - 입력 게이트
  - 출력 게이트

#### 입력 게이트
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/53b58d43-5d92-49de-bc49-3f789413c58e" />

현재 정보를 기억하기 위한 게이트

#### 삭제 게이트
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/7ee060be-eb1c-4f28-9546-7aa77002b842" />

기억을 삭제하기 위한 게이트

#### 셀 상태
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/739483ed-3aa0-466c-96d3-abcbc4adb0f3" />

입력 게이트에서 선택된 기억을 삭제 게이트의 결과값과 더함. 이 값을 현재 시점 t의 셀 상태라고 하며, 이 값은 다음 t+1 시점의 LSTM 셀로 넘겨짐<br>
결과적으로 삭제 게이트는 이전 시점의 입력을 얼마나 반영할지를 의미하고, 입력 게이트는 현재 시점의 입력을 얼마나 반영할지를 결정

#### 출력 게이트와 은닉 상태
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/d8eda569-cce5-4ae9-9105-ac17e9ec460f" />

출력 게이트는 현재 시점 t의 x값과 이전 시점 t-1의 은닉 상태가 시그모이드 함수를 지난 값<br>
해당 닶은 현재 시점 t의 은닉 상태를 결정하는 일에 쓰임<br>
셀 상태의 값은 하이볼릭탄젠트 함수를 지나 -1과 1사이의 값이 되고, 해당 값은 출력 게이트의 값과 연산되면서, 값이 걸러지는 효과가 발생하여 은닉 상태가 됨<br> 
은닉 상태의 값은 또한 출력층으로도 향함


## 8-3 게이트 순환 유닛(Gated Recurrent Unit, GRU)
GRU는 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트 하는 계산을 줄임<br>
즉, 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 간단화 함
### GRU(Gated Recurrent Unit)
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/a0820d08-b760-406b-9a3e-1d256a5e0ea5" />

- LSTM에는 출력, 입력, 삭제의 3개의 게이트가 존재했지만, GRU는 업데이트 게이트와 리셋 게이트두가지만 존재
- LSTM보다 학습 속도가 빠르지만 성능은 비슷함
- 데이터 양이 적을 때는 매개 변수의 양이 적은 GRU가 조금 더 낫고, 데이터 양이 많으면 LSTM이 더 나음


## 8-4 케라스의 SimpleRNN과 LSTM 이해하기
### 임의의 입력 생성하기
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional

# 벡터의 차원 = 5, 문장의 길이 = 4
### RNN은 2D 텐서가 아니라 3D 텐서를 입력받으므로 배치 크기 1을 추가함으로서 2D 텐서를 3D 텐서로 변경
train_X = [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]
print(np.shape(train_X))
```

### SimpleRNN 이해하기
```python 
# 은닉 상태의 크기를 3으로 지정하고, 두 인자 값이 모두 False일 때의 출력값 확인
rnn = SimpleRNN(3)
# rnn = SimpleRNN(3, return_sequences=False, return_state=False)와 동일.
hidden_state = rnn(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))

# return_sequences를 True로 지정하여 모든 시점의 은닉 상태를 출력
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))

#return_state가 True일 경우에는 return_sequences의 True/False 여부와 상관없이 마지막 시점의 은닉 상태를 출력
rnn = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_state = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))

# return_sequences는 False인데, retun_state가 True인 경우
rnn = SimpleRNN(3, return_sequences=False, return_state=True)
hidden_state, last_state = rnn(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
```

### LSTM 이해하기
```python
# return_sequences를 False로 두고, return_state가 True인 경우
lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))

# return_sequences를 True로 변경
lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_hidden_state, last_cell_state = lstm(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'.format(last_hidden_state, last_hidden_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
# return_state가 True이므로 두번째 출력값이 마지막 은닉 상태, 세번째 출력값이 마지막 셀 상태인 것은 변함없지만 return_sequences가 True이므로 첫번째 출력값은 모든 시점의 은닉 상태가 출력됨
```

### Bidirectional(LSTM) 이해하기
```python
# 양방향 LSTM의 출력값 확인

# 출력되는 은닉 상태의 값 고정
k_init = tf.keras.initializers.Constant(value=0.1)
b_init = tf.keras.initializers.Constant(value=0)
r_init = tf.keras.initializers.Constant(value=0.1)

# return_sequences가 False이고, return_state가 True인 경우
bilstm = Bidirectional(LSTM(3, return_sequences=False, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))
```
- 5개의 값을 반환
- 셀 상태는 각각 forward_c와 backward_c에 저장만 하고 출력하지 않음
- 정방향 LSTM의 마지막 시점의 은닉 상태값 : [0.6303139 0.6303139 0.6303139]
- 역방향 LSTM의 첫번째 시점의 은닉 상태값 : [0.70387346 0.70387346 0.70387346]
```python
# return_sequences를 True로 할 경우, 출력이 어떻게 바뀌는지 비교
bilstm = Bidirectional(LSTM(3, return_sequences=True, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)
print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))
```
- hidden states의 출력값에서는 이제 모든 시점의 은닉 상태가 출력
- 역방향 LSTM의 첫번째 시점의 은닉 상태는 더이상 정방향 LSTM의 마지막 시점의 은닉 상태와 연결되는 것이 아니라 정방향 LSTM의 첫번째 시점의 은닉 상태와 연결됨


## 8-5 RNN 언어 모델(Recurrent Neural Network Language Model, RNNLM)
### RNN 언어 모델(Recurrent Neural Network Language Model, RNNLM)
- n-gram 언어 모델과 NNLM은 고정된 개수의 단어만을 입력 받아야 한다는 단점이 존재
- 시점(time step) 개념이 도입된 RNN으로 언어 모델을 만들면 입력의 길이를 고정하지 않아도 됨
- RNN으로 만든 언어 모델을 RNNLM이라 함
#### RNNLM이 언어 모델링을 학습하는 과정
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/df5ab9b4-d093-4c6d-8de3-22746278e785" />

- RNNLM은 기본적으로 예측 과정에서 이전 시점의 출력을 현재 시점의 입력으로 함
- 교사 강요(teacher forcing)
  - RNN 훈련 기법
  - 테스트 과정에서 t 시점의 출력이 t+1 시점의 입력으로 사용되는 RNN 모델을 훈련시킬 때 사용하는 훈련 기법
  - 모델이 t 시점에서 예측한 값을 t+1 시점에 입력으로 사용하지 않고, t 시점의 레이블. 즉, 실제 알고있는 정답을 t+1 시점의 입력으로 사용
  - 교사 강요를 사용하여 RNN을 좀 더 빠르고 효과적으로 훈련시킬 수 있음
  - 훈련 과정 동안 출력층에서 사용하는 활성화 함수는 소프트맥스 함수
  - 모델이 예측한 값과 실제 레이블과의 오차를 계산하기 위해서 손실 함수로 크로스 엔트로피 함수를 사용
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/dd02549d-99f4-4e3c-b52a-74b7f6c9a7e3" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/14d800dc-3eab-4b14-8499-ac1669da1427" />

- RNNLM은 위의 그림과 같이 총 4개의 층(layer)으로 이루어진 인공 신경망
- timestep을 4로 가정한다면, 4번째 입력 단어인 fat의 원-핫 벡터가 입력됨
- 모델이 예측해야 하는 정답 단어인 cat의 원-핫 벡터는 출력층에서 모델이 예측한 값의 오차를 구하기 위해 사용됨
- 이 오차로부터 손실 함수를 사용해 인공 신경망이 학습함
- 임베딩층: 임베딩 벡터를 얻는 투사층


## 8-6 RNN을 이용한 텍스트 생성(Text Generation using RNN)
다대일 구조의 RNN을 사용하여 문맥을 반영해서 텍스트를 생성하는 모델 만들기

### RNN을 이용하여 텍스트 생성하기
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text = """경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

# 단어 집합을 생성하고 크기를 확인, 패딩을 위한 0을 고려하여 +1 처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)

# 훈련 데이터 만들기
sequences = list()
for line in text.split('\n'): # 줄바꿈 문자를 기준으로 문장 토큰화
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플의 개수: %d' % len(sequences))

# 전체 샘플에 대해 길이를 일치시키기
max_len = max(len(l) for l in sequences) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
print('샘플의 최대 길이 : {}'.format(max_len))

# 전체 샘플의 길이를 최대 길이로 패딩
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# 각 샘플의 마지막 단어를 레이블로 분리
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

# 원-핫 인코딩
y = to_categorical(y, num_classes=vocab_size)
```

### 모델 설계하기
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

# 하이퍼파라미터인 임베딩 벡터의 차원은 10, 은닉 상태의 크기는 32
# 다대일 구조의 RNN을 사용
# 전결합층(Fully Connected Layer)을 출력층으로 단어 집합 크기만큼의 뉴런을 배치하여 모델을 설계
embedding_dim = 10
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

# 모델이 정확하게 예측하고 있는지 문장을 생성하는 함수 생성 및 출력
def sentence_generation(model, tokenizer, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
        # 현재 단어에 대한 정수 인코딩과 패딩
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
        # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items(): 
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면 break
            if index == result:
                break

        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' '  + word

        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence
```

###  LSTM을 이용하여 텍스트 생성하기

#### 데이터에 대한 이해와 전처리

#### 모델 설계하기





