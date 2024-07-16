import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:\\Oasis\\Iris.csv")

print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.info)
print(df.describe())
print(df.columns)
print(df['Species'].value_counts())

sns.swarmplot(x="Species", y="SepalLengthCm",hue='PetalLengthCm', data=df)
plt.legend(title='Petal Length')
plt.show()

sns.boxplot(x="Species", y="SepalLengthCm",hue='PetalLengthCm', data=df)
plt.show()

sns.histplot(df['Species'], bins=10, kde=True)
plt.show()

corr = df.drop('Id', axis=1).select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title('Correlation Matrix Heatmap')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score

df['Species'] = df['Species'].astype('category').cat.codes

x = df.drop(['Id','Species'],axis=1)
y = df['Species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=51)
model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy Score is : ",accuracy_score(y_test,y_pred))

print("Classification Report is : ",classification_report(y_test,y_pred))




