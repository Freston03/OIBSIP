import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:\\Oasis\\spam.csv", encoding='latin-1')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps = PorterStemmer()
lemmetizer = WordNetLemmatizer()
corpus = []

for i in range(0,len(df)):
    changes = re.sub('[^a-zA-Z]', ' ', df['v2'][i])
    changes = changes.lower()
    changes = changes.split()

    changes = [ps.stem(word) for word in changes if not word in stopwords.words('english')]
    changes = ' '.join(changes)
    corpus.append(changes)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 3000)
X = vectorizer.fit_transform(corpus).toarray()
Y = pd.get_dummies(df['v1'])
Y = Y.iloc[:,1].values

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,r2_score,classification_report

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35,random_state=57)


model = MultinomialNB()
model = model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("Accuracy of the Model is : ",accuracy_score(Y_test,y_pred))
print("Classification Report is : ",classification_report(Y_test,y_pred))





