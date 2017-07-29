# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:55:18 2017

@author: tejpal.baghel
@Heckfest event 
"""
   
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt


#Loading data from source 
rawdataset = genfromtxt('processed.cleveland.data.csv',dtype = float, delimiter=',')
header= genfromtxt('header14.txt',dtype = str, delimiter=',')
df_data=  pd.DataFrame(rawdataset,columns= header.tolist())


# Data analyis - Get a quick idea of the data using stats based methods
print (df_data.shape)
df_data.head(5)
df_data.describe()

# Identify if there is any categorical features exists ,Must convert categorical features into numeric features
for column in df_data.columns:
    if df_data[column].dtypes == 'object':
        unique_count = len(df_data[column].unique())
        print("'{column}' feature have {unique_category} unique categories".format(column=column, unique_count=unique_count))

# Not exists, otherwise categorical features will requierd to convert in numbers using feature scaling and onehot encoding etc


# Dealing with missing data , how many nan etc ?
df_data.isnull().sum().sort_values(ascending=False).head()

# Impute missing values using Imputer in sklearn.preprocessing , retest missing values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(df_data)
df_data = pd.DataFrame(data=imp.transform(df_data) , columns=df_data.columns)

# retest missing values again to see if you still have missing data
df_data.isnull().sum().sort_values(ascending=False).head()


#Outlier detection if any
import OneplaceOntput
tukey_indices, tukey_values = find_outliers_tukey(df_data['chol'])

print(np.sort(tukey_values))

kde_indices, kde_values = find_outliers_kde(df_data['age'])
print(np.sort(kde_values))

# Split data in feture matrix X and output vector y
X= df_data.ix[:,:-1]
y= df_data.ix[:,-1:]

         
 

#Explore the data quickly by age , gender  etc


#Varius bar blots


#Box plots



#Histograms for age
import seaborn as sns
sns.set(color_codes=True)
sns.distplot(data["age"] )


# PCA   - diamension reduction techneque helpfull in case of large featureset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca=PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.title("Principal component analysis")
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
pca.score(X)

#Distribution on scatter plot

pca=PCA(2)
X_Projection=pca.fit_transform(X)
print(X.shape)
print(X_Projection.shape)
plt.legend(np.unique(y))
plt.scatter(X_Projection[:,0] , X_Projection[:,1] , c=y , alpha=1)
plt.colorbar()


#Prepar trainig and test dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2 , random_state=20) # random_state=20


####################################################
# Working with KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_prediction=knn.predict(X_test)
knn.score(X_test, y_test)

# Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn_prediction)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, knn_prediction))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, knn_prediction))

# calculate the fpr and tpr for all thresholds of the classification
from sklearn.metrics import roc_curve , auc ,roc_auc_score
knn_pp = knn.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, knn_pp)
roc_auc = auc(fpr, tpr)


# method plot ROC curve
import matplotlib.pyplot as plt
plt.hold(True)
plt.title('ROC - KNN')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.show()



print ("Work with KNN - output ")
OneplaceOntput.ClassificationAnalysis(y_test,knn_prediction)
knn_pp = knn.predict_proba(X_test)[:,1]
OneplaceOntput.plot_roc_curve(y_test,knn_pp,"ROC -KNN")



####################################################
# Working with MNB 
mnb=MultinomialNB()
mnb.fit(X_train , y_train  )
mnb_prediction=mnb.predict(X_test)
mnb.score(X_test, y_test)

from OneplaceOntput import OneplaceOntput
print ("Work with multiniminal - output ")
OneplaceOntput.ClassificationAnalysis( np.array(y_test) ,mnb_prediction)

mnb_pp = mnb.predict_proba(X_test)[:,1]
OneplaceOntput.plot_roc_curve(y_test,mnb_pp,"ROC -MultinomialNB")

####################################################
# Working with Gaussian
GNB=GaussianNB()
GNB.fit(X_train , y_train  )
GNB_prediction=mnb.predict(X_test)
GNB.score(X_test, y_test)

print ("Work with GaussianNB - output ")
OneplaceOntput.ClassificationAnalysis(np.array(y_test),GNB_prediction)
gnb_pp = mnb.predict_proba(X_test)[:,1]
OneplaceOntput.plot_roc_curve(y_test,gnb_pp,"ROC -MultinomialNB")


