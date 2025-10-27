import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
pd.set_option('display.max_columns', None)
data = pd.read_csv("COVID_Master_Tracker.csv")

#FIGURE OUT WHICH FEATURES TO SELECT
#data_restrict = data["Deaths", "BedsICU", "BedsAcute", "Beds"]
#print(data_restrict)

# Check for NA/improper datatypes
fieldnames= data.columns
print(data.dtypes)
NA_data = data.isna()

NA_data.to_csv('output_pandas.csv', index=False)
data['caseRCaro'] = data['caseRCaro'].fillna('None')
print("CASES MONTGOMERY NA:",data['caseRMont'].isna())
# don't need to check all columns, only
# Summary statistics
print(data.describe())



# plots
sns.scatterplot(y='caseRMont', x= 'deathGenMale', data =data)
plt.show()

"""
# apply ML models, knn, some classification, convolution? read amazon book
X = data["ReportDate"]
y = data["TotalCases"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2, stratify=y, random_state =69)

scaler = StandardScaler()
print(scaler.fit(data))
#decision tree"""