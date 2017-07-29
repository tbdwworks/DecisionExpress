# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:40:32 2017

@author: tejpal.baghel
"""

import numpy as np
from sklearn.metrics import roc_curve , auc , roc_auc_score

from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

class OneplaceOntput(object):
    @staticmethod
    def test ():
        print ("hi")
        return;
    @staticmethod
    def ClassificationAnalysis(actual_sentiment,knn_prediction):
        count=0
        for i in range (len(actual_sentiment)):
            if actual_sentiment[i]==knn_prediction[i]:
                count=count+1
        
        print ("Accurate pridiction {} out of {} ".format(count,len(actual_sentiment)))
        print ("Accuracy score is {} ".format(count/len(actual_sentiment)))
        
        # Accuracy score
        from sklearn.metrics import accuracy_score
        accuracy_score(actual_sentiment, knn_prediction)
        
        # Classification Report
        print ("Classification Report")
        from sklearn.metrics import classification_report
        print(classification_report(actual_sentiment, knn_prediction))
        
        #Confusion Matrix
        print ("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(actual_sentiment, knn_prediction))
        
        return
    def plot_roc_curve(y_test,model_pp,title):
        # calculate the fpr and tpr for all thresholds of the classification
        from sklearn.metrics import roc_curve , auc , roc_auc_score
        fpr, tpr, threshold = roc_curve(y_test, model_pp)
        roc_auc = auc(fpr, tpr) 
        roc_auc = roc_auc_score(y_test, model_pp)

        # method I: plt
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.show()
        # method II: plt
        #plt.plot(fpr,tpr,label="data 2, auc="+str(auc))
        #plt.legend(loc='best')
        return
    def multiple_roc_curve(y_test,model_tuple,title):
        # calculate the fpr and tpr for all thresholds of the classification
       
        # method I: plt
        import matplotlib.pyplot as plt
        plt.plot([0, 1], [0, 1],'--r')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        for i in range(len(model_tuple)):
            fpr, tpr, threshold = roc_curve(y_test, model_tuple[i][1])
            roc_auc = auc(fpr, tpr) 
            roc_auc = roc_auc_score(y_test, model_tuple[i][1])
            plt.title(title)
            #plt.plot(fpr, tpr, 'b', label =  "%s AUC = %f".format(model_tuple[i][0],roc_auc ))
            plt.plot(fpr, tpr, label =  model_tuple[i][0])
            
        plt.legend(loc = 'lower right')
        plt.show()
        return
    
    def find_outliers_tukey(x):
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3-q1 
        floor = q1 - 1.5*iqr
        ceiling = q3 + 1.5*iqr
        outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
        outlier_values = list(x[outlier_indices])
    
        return outlier_indices, outlier_values

    def find_outliers_kde(x):
        x_scaled = scale(list(map(float, x)))
        kde = KDEUnivariate(x_scaled)
        kde.fit(bw="scott", fft=True)
        pred = kde.evaluate(x_scaled)
        
        n = sum(pred < 0.05)
        outlier_ind = np.asarray(pred).argsort()[:n]
        outlier_value = np.asarray(x)[outlier_ind]
    
        return outlier_ind, outlier_value
