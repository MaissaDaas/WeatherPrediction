import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
url = "https://raw.githubusercontent.com/shravyapendyala/CCE_Assignment_1/refs/heads/main/weather_forecast_data.csv"
dataset=pd.read_csv(url)
dataset.head(10)

# Check for Missing Values
print("\nMissing Values in Each Column:")
print(dataset.isnull().sum())

# Label Encoding on Rain Column
label_encoder=preprocessing.LabelEncoder()
dataset['Rain']=label_encoder.fit_transform(dataset['Rain'])
dataset.head(5)

# Feature Selection using Correlation matrix
correl_mat=dataset.corr()
sb.heatmap(correl_mat,annot=True)

# Feature Selection using Extra Tree Classifier
x=dataset.iloc[:, 0:5]
y=dataset.iloc[:, -1]
model=ExtraTreesClassifier()
model.fit(x,y)
imp=pd.Series(model.feature_importances_)
feature_importance = model.feature_importances_
plt.bar(x.columns, feature_importance)

# Feature Selection using Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(x,y)
important_features = rf_model.feature_importances_

plt.figure(figsize=(6, 6))
plt.pie(important_features, labels=x.columns, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()

# Dimentionality reduction by removing Wind_speed column
dataset.columns
new_dataset=dataset.drop(columns=['Wind_Speed'])
new_dataset

# 
X=new_dataset.iloc[:, 0:4]
Y=new_dataset.iloc[:, -1]


