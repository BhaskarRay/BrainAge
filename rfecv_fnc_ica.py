# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 02:46:43 2020

@author: Bhaskar
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn import preprocessing

intersect_fnc_l = pd.read_csv("/data/mialab/users/bray14/rfecv_fnc/intersect_fnc_ica_2_filtered.csv")
X = intersect_fnc_l.iloc[:,5:105]
target = intersect_fnc_l['age_at_cnb']
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(X)
Svr_linear = SVR(kernel='linear')
rfecv = RFECV(estimator=Svr_linear, step=1, cv=StratifiedKFold(10), scoring='neg_root_mean_squared_error')
rfecv.fit(X_train_minmax, target)

print('Optimal number of features: {}'.format(rfecv.n_features_))


coef_rmse_rfecv= rfecv.estimator_.coef_
rfecv_rmse_featureCoeff = pd.DataFrame()
rfecv_rmse_featureCoeff['attr'] = X.columns[rfecv.support_]
rfecv_rmse_featureCoeff['coefficient'] = coef_rmse_rfecv.transpose(1,0)
rfecv_rmse_featureCoeff['rank']= rfecv.ranking_[rfecv.support_]
rfecv_rmse_featureCoeff = rfecv_rmse_featureCoeff.sort_values(by='coefficient', ascending=False)
rfecv_rmse_featureCoeff.to_csv('rfecv_rmse_featureCoeff.csv',encoding='utf-8',index=False,na_rep='NA')

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('Neg RMSE', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

#plt.show()
plt.savefig('/data/mialab/users/bray14/rfecv_fnc/rfecvFNCplot.png')

plt.figure(figsize=(16, 14))
plt.barh(y=rfecv_rmse_featureCoeff['attr'], width=rfecv_rmse_featureCoeff['coefficient'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
#plt.show()
plt.savefig('/data/mialab/users/bray14/rfecv_fnc/rfecvFNCplot2.png')