# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:29:49 2020

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


intersect_ICA_AAL_freeSrfr_fnc_l = pd.read_csv("/data/mialab/users/bray14/rfecv_fnc/intersect_ICA_AAL_freeSrfr_fnc_ica_filtered.csv")
intersect_ICA_AAL_freeSrfr_fnc_l.drop(["age_at_cnb_y",'Sex_y',"Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
X_all = intersect_ICA_AAL_freeSrfr_fnc_l.iloc[:,5:473]
mm_scaler = preprocessing.MinMaxScaler()
X_all_train_minmax = mm_scaler.fit_transform(X_all)
target_all = intersect_ICA_AAL_freeSrfr_fnc_l['age_at_cnb_x']
Svr_linear = SVR(kernel='linear')
rfecvall = RFECV(estimator=Svr_linear, step=1, cv=StratifiedKFold(10), scoring='r2')
#neg_root_mean_squared_error
#r2
rfecvall.fit(X_all_train_minmax, target_all)

print('Optimal number of features: {}'.format(rfecvall.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
#plt.ylabel('Neg RMSE', fontsize=14, labelpad=20)
plt.ylabel('R2 Score', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecvall.grid_scores_) + 1), rfecvall.grid_scores_, color='#303F9F', linewidth=3)

#plt.show()
plt.savefig('/data/mialab/users/bray14/rfecv_fnc/rfecvAllFNCplot.png')

coef_rmse_rfecv_all= rfecvall.estimator_.coef_
rfecv_rmse_featureCoeff_all = pd.DataFrame()
rfecv_rmse_featureCoeff_all['attr'] = X_all.columns[rfecvall.support_]
rfecv_rmse_featureCoeff_all['coefficient'] = coef_rmse_rfecv_all.transpose(1,0)
rfecv_rmse_featureCoeff_all['rank']= rfecvall.ranking_[rfecvall.support_]
rfecv_rmse_featureCoeff_all = rfecv_rmse_featureCoeff_all.sort_values(by='coefficient', ascending=False)
rfecv_rmse_featureCoeff_all.to_csv('rfecv_rmse_featureCoeff_all.csv',encoding='utf-8',index=False,na_rep='NA')
plt.barh(y=rfecv_rmse_featureCoeff_all['attr'], width=rfecv_rmse_featureCoeff_all['coefficient'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.savefig('/data/mialab/users/bray14/rfecv_fnc/rfecvALLFNCplot2.png')