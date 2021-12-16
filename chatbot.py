import random
import nltk #edit_distance how dissimilar two strings (e.g., words) are
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #converts lists to a matrix of nums ВЕКТОРАЙЗЕР
from sklearn.linear_model import LogisticRegression #machine learning model КЛАССИФИКАТОР
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.model_selection import train_test_split #splits data into test and training data РАЗБИЕНИЕ НА ТРЕНИРОВОЧНУЮ И ТЕСТОВУЮ
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
with open('/Users/Evgenia/Downloads/BOT_CONFIG.json') as f:
    BOT_CONFIG = json.load(f) #readfile


def clean(text):
    cleaned_text = ''
    for char in text.lower():
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя ':
            cleaned_text = cleaned_text + char
    return cleaned_text

corpus = []
y = [] #intents
for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
        corpus.append(example)
        y.append(intent)


corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.33, random_state=42)


vectorizer = CountVectorizer(ngram_range=(1,3), analyzer='char_wb', binary=False, max_features=900) #TfidfVectorizer() #prepocessor = clean
 #векторизуем тексты (на тренировочной создаем словарь, к тестовой только применяем
X_train = vectorizer.fit_transform(corpus_train) #makes a table, returns vectors, fit_transform - training
X_test =  vectorizer.transform(corpus_test) #transform - test data
vocabulary = vectorizer.get_feature_names() #returns dict СЛОВА, КОТОРЫЕ ВСТРЕТИЛИСЬ



 #the size of the matrix
#print(X.toarray()) #returns vectorized values
#print(dict_vectorizer)

model = RandomForestClassifier(n_estimators=500, min_impurity_decrease = 0) #LogisticRegression(C=0.7) N-estimators - деревья
model.fit(X_train,y_train) #model is studying учим на тренировочной части, точность 70

estimate_test = model.score(X_test,y_test) #accuracy, using test data for an estimate, точность 10
                        # предсказывает намерения в объектах, которые она не видела, точность 10
estimate_training = model.score(X_train, y_train)
print(estimate_training, estimate_test)

with open('/Users/Evgenia/Downloads/vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('/Users/Evgenia/Downloads/classifier.pickle', 'wb') as f:
    pickle.dump(model, f)


with open('/Users/Evgenia/Downloads/classifier.pickle', 'rb') as f:
    model = pickle.load(f)

def get_intent_by_model(text):
    return model.predict(vectorizer.transform([text]))[0] #index zero?


def get_intent(text):
    for intent in BOT_CONFIG['intents'].keys():
        for example in BOT_CONFIG['intents'][intent]['examples']:
            cleaned_text = clean(text)
            cleaned_example = clean(example)
            if nltk.edit_distance(cleaned_example,cleaned_text) / max(len(cleaned_example), len(cleaned_text)) < 0.4:
                return intent
    return 'intent not found'


def bot(text):
    intent = get_intent_by_model(text)
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])



while True:
  text = input()
  answer = bot(text)
  print(answer)


