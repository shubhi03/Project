# Created By Shubham

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/admin/Desktop/Data Science/01_DataScience/Datasets/Log-Reg-Case-Study.csv')

dataset.head()
# The head() is used to get the first 5 Rows of the dataset
dataset.tail()
# tail() is to fetch last 5 Rows
dataset.info()
# info() to get the information of the dataset we have loaded that if there are missing values or not
dataset.describe()
# to get the mathematical part of dataset i,e mean ,median ,mode ,etc...


# now dividing the data into Dependent and Independent Variable
X = dataset.iloc[:,[1,10,11,13]].values
y = dataset.iloc[:,16].values

#----------Capping(Handling the outliers)-----------------

dataset.Age.describe()
dataset.Age.quantile(q=0.995)
min(dataset.Age)
max(dataset.Age)


dataset.loc[dataset['Age']>75,'Age'] = 75
# Now

min(dataset.Age)
max(dataset.Age)
# Now our Outlier present in Age Column is handled & max Age came to 75

# Missing Values
dataset.isnull().values.any()

dataset.isnull().sum()
# so by this we know that there are 12 values missing in Job_Status & 9 in Housing

# Housing
dataset['Housing'].mode()

dataset.Housing[dataset.Housing=='A151'].count()
dataset.Housing[dataset.Housing=='A152'].count()
dataset.Housing[dataset.Housing=='A153'].count()

dataset['Housing'].fillna(dataset['Housing'].mode()[0],inplace = True)

dataset.Housing[dataset.Housing=='A152'].count()

dataset.Housing.isnull().sum()

# For Job_Status

dataset['Job_Status'].isnull().sum()

dataset['Job_Status'].describe()
dataset['Job_Status'].unique()

dataset.Job_Status.value_counts()
dataset['Job_Status'].mode()

pd.crosstab(dataset.Job_Status,dataset.Default_On_Payment)
dataset['Job_Status'].fillna(dataset['Job_Status'].mode()[0],inplace = True)

dataset['Job_Status'].describe()
dataset['Job_Status'].isnull().sum()

# Now
dataset.info()
dataset.isnull().sum().sum()

# Now splitting the data to training and testing part
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3, random_state=0)


# Just importing StandardScaler for feature Scaling from sklearn to Get The values in the range of +1 to -1 and  prediction becomes High.
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# importing Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Importing confusion_matrix To check the Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Importing accuracy_score To check the Accuracy Score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)



# So By Checking the AccuracyScore & The Confusion Matrix it is Clear That Our Model is Good But Can be Improved by Working More And by Changing the Training and Testing Part.

....................................................The End...................................................
