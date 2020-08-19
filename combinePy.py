import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

intersect = pd.read_csv("intersectData_CSV1.csv")
pd.set_option('display.max_columns', None)
intersect
intersect.drop(['Unnamed: 0'],axis=1)
intersect.drop(['	WRAT_VALID'],axis=1)
intersect.drop(["SIP001","SIP002","SIP003","SIP004","SIP005","SIP006","SIP007","SIP008","SIP009","SIP010","SIP011","SIP012","SIP013","SIP014","SIP015","SIP016","SIP017","SIP018","SIP019","SIP020","SIP021","SIP022","SIP023","SIP024","SIP025","SIP026","SIP027","SIP028","SIP030","SIP031","SIP032","SIP033","SIP035","SIP036","SIP037","SIP038","SIP039","SIP041","SIP042","SIP043","SIP044"],axis=1)
intersect.drop(["SIP001","SIP002","SIP003","SIP004","SIP005","SIP006","SIP007","SIP008","SIP009","SIP010","SIP011","SIP012","SIP013","SIP014","SIP015","SIP016","SIP017","SIP018","SIP019","SIP020","SIP021","SIP022","SIP023","SIP024","SIP025","SIP026","SIP027","SIP028","SIP030","SIP031","SIP032","SIP033","SIP035","SIP036","SIP037","SIP038","SIP039","SIP041","SIP042","SIP043","SIP044","	WRAT_VALID",'Unnamed: 0',"WRAT_CR_RAW","WRAT_CR_STD","INT_TYPE"],axis=1,inplace=True)
intersect

intersect.to_csv('intersect.csv',encoding='utf-8',na_rep='NA')
ICA = pd.read_csv("ICAdataSetOrPresn.csv",header=None)
list=[]
for i in range(1,101):
    list.append("ICA_COMP_"+str(i))

	
ICA.columns=list
ICA
ICA['SUBJID'] = intersect['SUBJID'].values
ICA


ICA.to_csv('ICA_col.csv',encoding='utf-8', index=False, na_rep='NA')
ICA.reset_index(drop=True, inplace=True)
ICA


ICA.to_csv('ICA_col.csv',encoding='utf-8', index=False, na_rep='NA')
intersect_ICA=pd.merge(intersect,ICA, how='inner', on=['SUBJID'])
intersect_ICA


AAL = pd.read_csv("AAL_116_RoidataSetOrPresn.csv",header=None)

list_AAL=[]
for i in range(1,117):
    list_AAL.append("AAL_ROI_"+str(i))
	
AAL.columns=list_AAL

AAL['SUBJID'] = intersect['SUBJID'].values
AAL


intersect_ICA_AAL=pd.merge(intersect_ICA,AAL, how='inner', on=['SUBJID'])
intersect_ICA_AAL
freeSrfr = pd.read_csv("freeSrfr_Med_SIP_ordered.csv",index_col=False)
freeSrfr


freeSrfr.drop(["SIP001","SIP002","SIP003","SIP004","SIP005","SIP006","SIP007","SIP008","SIP009","SIP010","SIP011","SIP012","SIP013","SIP014","SIP015","SIP016","SIP017","SIP018","SIP019","SIP020","SIP021","SIP022","SIP023","SIP024","SIP025","SIP026","SIP027","SIP028","SIP030","SIP031","SIP032","SIP033","SIP035","SIP036","SIP037","SIP038","SIP039","SIP041","SIP042","SIP043","SIP044","	WRAT_VALID",'Unnamed: 0',"WRAT_CR_RAW","WRAT_CR_STD","INT_TYPE","Unnamed: 0","Unnamed: 0.1","age_at_cnb","Sex","Med_Rating","SIP_COUNT"],axis=1,inplace=True)
freeSrfr
intersect_ICA_AAL_freeSrfr=pd.merge(intersect_ICA_AAL,freeSrfr, how='inner', on=['SUBJID'])
intersect_ICA_AAL_freeSrfr

intersect_ICA_AAL_freeSrfr['Med_Rating'] = intersect_ICA_AAL_freeSrfr['Med_Rating'].fillna(0)
intersect_ICA_AAL_freeSrfr
intersect_ICA_AAL_freeSrfr.drop(intersect_ICA_AAL_freeSrfr[intersect_ICA_AAL_freeSrfr['SIP_COUNT'] <=14].index, inplace = True)
intersect_ICA_AAL_freeSrfr
intersect_ICA_AAL_freeSrfr.drop(intersect_ICA_AAL_freeSrfr[intersect_ICA_AAL_freeSrfr['Med_Rating']>2].index, inplace = True)
intersect_ICA_AAL_freeSrfr


intersect_ICA_AAL_freeSrfr2=intersect_ICA_AAL_freeSrfr
intersect_ICA_AAL_freeSrfr2.drop(["SUBJID","Sex","Med_Rating","SIP_COUNT"],axis=1,inplace=True)
intersect_ICA_AAL_freeSrfr2
intersect_ICA_AAL_freeSrfr2.to_csv('intersect_ICA_AAL_freeSrfr2.csv',encoding='utf-8',index=False,na_rep='NA')


correlated_features = set()
correlation_matrix = intersect_ICA_AAL_freeSrfr2.drop('age_at_cnb', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
			
pd.set_option('display.max_rows', None)
correlation_matrix

print(len(correlated_features))
intersect_ICA_AAL_freeSrfr2.drop(correlated_features, axis=1,inplace=True)
intersect_ICA_AAL_freeSrfr2.to_csv('intersect_ICA_AAL_freeSrfr2_reduced.csv',encoding='utf-8',index=False,na_rep='NA')
intersect_ICA_AAL_freeSrfr2.shape

X = intersect_ICA_AAL_freeSrfr2.drop('age_at_cnb', axis=1)
target = intersect_ICA_AAL_freeSrfr2['age_at_cnb']
Svr_linear = SVR(kernel='linear')
rfecv = RFECV(estimator=Svr_linear, step=1, cv=StratifiedKFold(10), scoring='neg_root_mean_squared_error')
rfecv.fit(X, target)


X = intersect_ICA_AAL_freeSrfr2.drop('age_at_cnb', axis=1)
target = intersect_ICA_AAL_freeSrfr2['age_at_cnb']
Svr_linear = SVR(kernel='linear')
rfe = RFE(estimator=Svr_linear, step=1)
rfe = rfe.fit(X,target)