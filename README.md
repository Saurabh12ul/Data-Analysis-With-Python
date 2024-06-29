#Importing Important Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder




#Checking Data Info
df.shape
df.isnull().sum()
df["Price"].value_counts()
df["Model Name"].value_counts()
df["Engine Type"].value_counts()




#Plotting Graph
sns.countplot(x="Engine Type",data=df)
plt.show()
#Calculate And Remove Empty Places
miss_per=df.isnull().sum()/df.shape[0]
miss_per
for column in df.columns:
    df[column]=df[column].fillna(df[column].mode()[0])
    for column in df.columns:
    if df[column].dtype==object:
        df[column]=LabelEncoder().fit_transform(df[column])
        x=df.drop("Price",axis=1)
        y=df["Price"]
type(y)
y=y.to_frame()
type(y)




#Import Train test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#Train and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=25)




#Applying Model
model=LinearRegression()
model.fit(x_train,y_train)
x_test_pred=model.predict(x_test)



#Calculating Accuracy of data
accu = metrics.r2_score(y_test,x_test_pred)
print("Accuracy : ",accu*100,"%")
x_train_pred=model.predict(x_train)
accu = metrics.r2_score(y_train,x_train_pred)
print("Accuracy : ",accu*100,"%")
