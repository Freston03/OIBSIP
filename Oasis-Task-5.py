import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:\\Oasis\\Advertising.csv")

print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.info)
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df.isna().sum())

sns.barplot(x='TV', y='Newspaper', data=df)
plt.xticks(rotation=90)
plt.title('TV and Newspaper Analysis')
plt.show()

plt.hist(df['TV'],bins=10, color="blue")
plt.title('TV Analysis')
plt.show()

plt.hist(df['Radio'],bins=10, color="green")
plt.title('Radio Analysis')
plt.show()

plt.hist(df['Newspaper'],bins=10, color="purple")
plt.title('Newspaper Analysis')
plt.show()

plt.hist(df['Sales'],bins=10, color="red")
plt.title('Sales Analysis')
plt.show()

sns.heatmap(df.corr(),annot = True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

X = df[['TV']]
Y = df[['Sales']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35,random_state=79)

model = LinearRegression()
model = model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("R2 Score  of the Model is : ",r2_score(y_pred,Y_test))
print("Mean Squared Error of report is : ",mean_squared_error(y_pred,Y_test))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='green', label='Actual Data')
plt.plot(X_test, y_pred, color='magenta', label='Predicted Data')
plt.title('Scatter Plot Comparison')
plt.xlabel('TV')
plt.ylabel('Newspaper')
plt.legend()
plt.grid(True)
plt.show()
