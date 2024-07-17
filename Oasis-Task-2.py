import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:\\Oasis\\Unemployment_Rate.csv")

print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.info)
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df.isna().sum())

df.rename(columns={'Region': 'State In India'}, inplace=True)
average_rate_of_unemployment = df.groupby('State In India')[' Estimated Unemployment Rate (%)'].mean()

highest_unemployment_state = average_rate_of_unemployment.idxmax()
highest_unemployment = average_rate_of_unemployment.max()

print("State with the highest unemployment rate:", highest_unemployment_state)
print("Highest unemployment rate:", highest_unemployment)


lowest_unemployment_state = average_rate_of_unemployment.idxmin()
lowest_unemployment = average_rate_of_unemployment.min()

print("State with the highest unemployment rate:", lowest_unemployment_state)
print("Highest unemployment rate:", lowest_unemployment)

#Bar-Chart
sns.barplot(x='State In India', y=' Estimated Unemployment Rate (%)', data=df)
plt.xticks(rotation=90)
plt.title('Average Unemployment Rate by State')
plt.show()

#Box-Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='State In India', y=' Estimated Unemployment Rate (%)', data=df)
plt.xticks(rotation=90)
plt.title('Distribution of Unemployment Rates by State')
plt.xlabel('State In India')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.show()

#Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df[' Estimated Unemployment Rate (%)'], bins=20, kde=True)
plt.title('Distribution of Estimated Unemployment Rate (%)')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()