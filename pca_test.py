#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:58:44 2020

@author: bray14
"""

import numpy as np
from random import *
from sklearn import preprocessing
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.decomposition import PCA
###### List of Classifier 
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


NO_OF_SPLITS = [10]
SEED = randint(1, 9999999)


classifier_dictionary = [
    
    ["SVR_LINEAR", make_pipeline(SVR(kernel='linear')), "SVR kernel=linear"], 
    ["PLSRegression", make_pipeline(PLSRegression(copy=True, max_iter=500, n_components=5, scale=True,tol=1e-06)), "PLSRegression"]
    
]




def trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile):
    givenTestFeatures = testData   
    accuracyVector = []
             

    for i in classifier_dictionary:    
        bestAccuracyAmongKFold = 0
        try:            
            for k in NO_OF_SPLITS:
                kFold = StratifiedKFold(n_splits = k, shuffle = True, random_state = SEED)
                
                print(i[0])
                classifier = i[1]
                scores = cross_val_score(classifier, trainData, trainLabel, cv = kFold, scoring='neg_root_mean_squared_error')
                
                

                print(scores)                      
                    
                print("Accuracy with scaled data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                
                bestAccuracyAmongKFold = max(scores.mean(), bestAccuracyAmongKFold)                                
                        
            
            #print(bestAccuracyAmongKFold)            
        except IOError:
            print('An error occurred trying to read the file.')

        except ValueError:
            print('Non-numeric data found in the file.')

        except ImportError:
            print ("NO module found")

        except EOFError:
            print('EOFError')

        except KeyboardInterrupt:
            print('You cancelled the operation.')

        accuracyVector.append(bestAccuracyAmongKFold)

    #print("accuracyVector :: ")
    #print(accuracyVector)
    
    #print("highest accurate classifier :: ")
    
    #print(classifier_dictionary[np.argmax(accuracyVector, axis=0)][0])
    #print("highest accuracy :: ")
    #print(np.amax(accuracyVector, axis=0))
    #mostAccurateClassifier = classifier_dictionary[np.argmax(accuracyVector, axis=0)][1]
  

    
    print(classifier_dictionary[0][0])
    classifier_dictionary[0][1].fit(trainData, trainLabel)
    preditedLabel_linear = classifier_dictionary[0][1].predict(givenTestFeatures)
    r2Score=r2_score(predictedLabelFile,preditedLabel_linear)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(predictedLabelFile, preditedLabel_linear)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    
    
    print(classifier_dictionary[1][0])
    classifier_dictionary[1][1].fit(trainData, trainLabel)
    preditedLabel_PLS = classifier_dictionary[1][1].predict(givenTestFeatures)
    r2Score_PLS=r2_score(predictedLabelFile,preditedLabel_PLS)
    print("PLS R^2 Score:", r2Score_PLS)
    mse_PLS =mean_squared_error(predictedLabelFile, preditedLabel_PLS)
    print("PLS Mean Squared Error:",mse_PLS)
    rmse_PLS = math.sqrt(mse_PLS)
    print("PLS Root Mean Squared Error:", rmse_PLS)


    plt.figure(figsize=(16, 9))
    data_test = {'ChronologicalAge': predictedLabelFile,'BrainAge':preditedLabel_linear}
    resultdf = pd.DataFrame(data_test)
    #sns.regplot(x='ChronologicalAge',y='BrainAge', data=resultdf)
    sns.lmplot( x='ChronologicalAge',y='BrainAge', data=resultdf, x_jitter=0.30)



data = pd.read_csv("/data/mialab/users/bray14/RFE/intersect_ICA_AAL_freeSrfr2.csv")
X_dataset = data.iloc[:,1:369]
Y_dataset = data.iloc[:,0]

scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = RobustScaler()

scaler.fit(X_dataset)
X_data_standard = scaler.transform(X_dataset)





#pca = PCA(n_components=2)
pca = PCA(.95)
#pca = PCA(.90)
#pca = PCA(.80)

pca.fit(X_data_standard)

print("Number of PCA components: ", pca.n_components_)
#print("PCA explained variance: ", pca.explained_variance_)
print("PCA explained variance ratio:\n", pca.explained_variance_ratio_)
print("Cumulative sum of PCA explained variance ratio:\n",np.cumsum(pca.explained_variance_ratio_))


plt.figure(figsize=(16, 9))
plt.title('PCA explained variance ratio', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of PCA component', fontsize=14, labelpad=20)
plt.ylabel('Cumulative sum of PCA explained variance ratio', fontsize=14, labelpad=20)
plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), color='#303F9F', linewidth=3)


X_data_standard_pca = pca.transform(X_data_standard)
df=pd.DataFrame(data=X_data_standard_pca[0:,0:],index=[i for i in range(X_data_standard_pca.shape[0])],columns=['PCA_COMP'+str(i) for i in range(X_data_standard_pca.shape[1])])
df.to_csv('pca.csv',encoding='utf-8',na_rep='NA')
X_data_standard.to_csv('X_data_PCA.csv',encoding='utf-8',na_rep='NA')
Y_dataset.to_csv('Y_dataset.csv',encoding='utf-8',na_rep='NA')

X_train, X_test, y_train, y_test = train_test_split(X_data_standard_pca, Y_dataset, test_size = 0.10)
trainModelWithDataset(X_train, y_train, X_test, y_test)