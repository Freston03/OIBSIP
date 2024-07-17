import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:\\Oasis\\Car_data.csv")

print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.info)
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df.isna().sum())


plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Name', y='Driven_kms', data=df)
plt.xticks(rotation=90)
plt.title('Average Distance travelled by Brand')
plt.xlabel('Brand/Car Name')
plt.ylabel('Average Distance Travelled')
plt.show()

df = pd.get_dummies(df, drop_first=True)
X = df.drop('Present_Price', axis=1)
Y = df['Selling_Price']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=67)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error is : ",mean_absolute_error(Y_test,y_pred))
print("Mean Square Error is : ",mean_squared_error(Y_test,y_pred))

root_mean_squared = mean_squared_error(Y_test,y_pred) ** 0.5
print("Root Mean Squared Error is : ",root_mean_squared)

print("R2 Score is : ",r2_score(Y_test,y_pred))

