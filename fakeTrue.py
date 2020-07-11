'''
Can you use this data set to make an algorithm able to determine if an article is fake news or not ?

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score,roc_auc_score,roc_curve

fake = pd.read_csv('e:/Fake.csv')
true = pd.read_csv('e:/True.csv')

print('null values in Fake',fake.isnull().sum())
print('null values in True',true.isnull().sum())

plt.figure(figsize=(12,6))
sns.countplot(x='subject',data=fake)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='subject',data=true)
plt.show()

true['category'] = 1
fake['category'] = 0
df = pd.concat([true,fake])
print('Shape',df.shape)

sns.countplot(df.category)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(df.subject)
plt.show()

#Combining Features

df['combined_text'] = df['text'] + ' ' + df['title'] + ' ' +df['subject']
del df['title']
del df['text']
del df['subject']
del df['date']

# removing unuseful words
STOPWORDS = set(stopwords.words('english'))
punctuations = string.punctuation
STOPWORDS.update(punctuations)


# Cleaning the text

def clean_text(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    text = re.sub(r'https?://\S+|www\.\S+',r'',text)
    text = re.sub('[\d]',r'',text)
    text = re.sub('[()]',r'',text)
    text = re.sub(r'(<.*?>)',r'',text)
    text = re.sub(r'[^(A-Za-z)]',r' ',text)
    text = re.sub(r'\s+',r' ',text)
  
    return text  

df['text'] = df['combined_text'].apply(lambda x : clean_text(x))

print(df[:10])

# Word Distribution between True and Fake Words

fake_len = df[df.category == 0].text.str.len()
true_len = df[df.category == 1].text.str.len()

plt.hist(fake_len, bins=20, label="fake_length")
plt.hist(true_len, bins=20, label="true_length")
plt.legend()
plt.show()

X = df.text
y = df.category

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

vect = CountVectorizer()
#training data
X_train = vect.fit_transform(x_train)

model = MultinomialNB()
history = model.fit(X_train,y_train)

print(history.score(X_train,y_train))
y_train_pred = cross_val_predict(model,X_train, y_train, cv=10)
print(y_train_pred)

roc_auc_score = roc_auc_score(y_train,y_train_pred)
print('roc_auc_score',roc_auc_score)

