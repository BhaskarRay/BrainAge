import numpy as np
from random import *
from sklearn import preprocessing

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA
from statistics import mean
###### List of Classifier 
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import RobustScaler
from scipy import stats


#NO_OF_SPLITS = [10]
NO_OF_SPLITS = [10]
SEED = randint(1, 9999999)


#neg_root_mean_squared_error
#r2
#MinMaxScaler()
#StandardScaler()

classifier_dictionary = [

    #["SVR_RBF", make_pipeline(preprocessing.MinMaxScaler(),SVR(kernel='rbf', C=1e4, gamma=0.1)), "SVR kernel=RBF"],
    ["SVR_LINEAR", make_pipeline(preprocessing.MinMaxScaler(), SVR(kernel='linear')), "SVR kernel=linear"],
    #["LinearSVR", make_pipeline(LinearSVR(max_iter = 2000) ), "LinearSVR"],
    #["SVR_POLYNOMIAL", make_pipeline(preprocessing.StandardScaler(),SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)), "SVR kernel=Polynomial"],
    ["PLSRegression", make_pipeline(preprocessing.MinMaxScaler(),PLSRegression(copy=True, max_iter=500,n_components=5,scale=True,tol=1e-06)), "PLSRegression"]
    #["PLSRegression", make_pipeline(preprocessing.MinMaxScaler(),PLSRegression(copy=True, max_iter=500,scale=True,tol=1e-06)), "PLSRegression"]
    #["PLSRegression", make_pipeline(preprocessing.MinMaxScaler(),PLSRegression(copy=True, max_iter=500, n_components=1911, scale=True,tol=1e-06)), "PLSRegression"]
    #["PLSRegression", make_pipeline(preprocessing.MinMaxScaler(),PLSRegression(copy=True, max_iter=500, n_components=5, scale=True,tol=1e-06)), "PLSRegression"]
    #["PLSRegression", make_pipeline(preprocessing.MinMaxScaler(),PLSRegression(copy=True, max_iter=500, n_components=5, scale=True,tol=1e-06)), "PLSRegression"]
       
]


#make_pipeline(preprocessing.MinMaxScaler(), SVR(kernel='linear'))
pipe=Pipeline(steps=[('MinMaxScaler', MinMaxScaler()),
                ('SVR_LINEAR', SVR(kernel='linear'))])

def best_fit_slope_and_intercept(xs,ys):
    
    #slope m
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    
    #intercept mean(y)-m*mean(x)
    b = mean(ys) - m*mean(xs)
    
    return m, b


#x_train_full,x_test_full
    
#x_full, y_full,x_full_feature,x_full_unhealthy,y_full_unhealthy,x_full_unhealthy_feature

def trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile):
    givenTestFeatures = testData
    #testDataPercentage = 0.4 # 40%     
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
                
                #bestAccuracyAmongKFold = max(scores.mean(), bestAccuracyAmongKFold)                                
                        
            
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
    
    '''
    print(classifier_dictionary[0][0])
    classifier_dictionary[0][1].fit(trainData, trainLabel)
    preditedLabel_rbf = classifier_dictionary[0][1].predict(givenTestFeatures)
    r2Score=r2_score(predictedLabelFile,preditedLabel_rbf)
    print("RBF SVR R^2 Score:", r2Score)
    mse =mean_squared_error(predictedLabelFile, preditedLabel_rbf)
    print("RBF SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("RBF SVR Root Mean Squared Error:", rmse)
    '''
    
    print(classifier_dictionary[0][0])
    classifier_dictionary[0][1].fit(trainData, trainLabel)
    preditedLabel_linear = classifier_dictionary[0][1].predict(givenTestFeatures)
    r2Score=r2_score(predictedLabelFile,preditedLabel_linear)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(predictedLabelFile, preditedLabel_linear)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    
    '''
    preditedLabel_linear_2 = classifier_dictionary[0][1].predict(X_test_male_FM)
    r2Score=r2_score(y_test_male_FM,preditedLabel_linear_2)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(y_test_male_FM, preditedLabel_linear_2)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    '''
    
    '''
    male_result = pd.DataFrame()
    male_result['Chronological Age'] = predictedLabelFile
    male_result['Brain Age(male model)'] = preditedLabel_linear
    male_result['Sex']='Male'
    
    female_result = pd.DataFrame()
    female_result['Chronological Age'] = y_test_female_MM
    female_result['Brain Age(male model)'] = preditedLabel_linear_2
    female_result['Sex'] = 'Female'
    '''
    '''
    female_result = pd.DataFrame()
    female_result['Chronological Age'] = predictedLabelFile
    female_result['Brain Age(female model)'] = preditedLabel_linear
    female_result['Sex'] = 'Female'
    
    male_result = pd.DataFrame()
    male_result['Chronological Age'] = y_test_male_FM
    male_result['Brain Age(female model)'] = preditedLabel_linear_2
    male_result['Sex']='Male'
    
    female_result=female_result.append(male_result, ignore_index=True)
    female_result.to_csv("F:/Pheno_data/dataSetFinalV/plot2/female_result.csv",index=True)
    '''
    
    '''
    male_result=male_result.append(female_result, ignore_index=True)
    male_result.to_csv("F:/Pheno_data/dataSetFinalV/plot2/male_result.csv",index=True)
    '''
    
    '''
    data_plot = pd.read_csv("F:/Pheno_data/dataSetFinalV/plot2/male_result.csv")
    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_style("ticks")
    ax = sns.lmplot(x='Chronological Age',y='Brain Age(male model)',hue='Sex',x_jitter=.50, data=data_plot,scatter_kws={'s':20})
    ax = (ax.set_axis_labels("Chronological Age","Brain Age (male model)").set(xlim=(8,22),ylim=(8,22)))
    ax.savefig('F:/Pheno_data/dataSetFinalV/plot2/male_model.png',dpi=300)
    '''
    
    '''
    data_plot = pd.read_csv("F:/Pheno_data/dataSetFinalV/plot2/female_result.csv")
    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_style("ticks")
    ax = sns.lmplot(x='Chronological Age',y='Brain Age(female model)',hue='Sex',x_jitter=.50, data=data_plot,scatter_kws={'s':20})
    ax = (ax.set_axis_labels("Chronological Age","Brain Age (female model)").set(xlim=(8,22),ylim=(6,22)))
    ax.savefig('F:/Pheno_data/dataSetFinalV/plot2/female_model.png',dpi=300)
    '''
    
    
    '''
    
    ##Full Model
    print("\n\nFull Model")
    print(classifier_dictionary[0][0])
    
    pipe.fit(x_full, y_full)
    preditedLabel_linear_full = pipe.predict(x_full)
    
    print("Intercept: ",pipe.named_steps['SVR_LINEAR'].intercept_)
    print("Coefficient: ",pipe.named_steps['SVR_LINEAR'].coef_)
    r2Score=r2_score(y_full,preditedLabel_linear_full)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(y_full, preditedLabel_linear_full)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    '''
    
    '''
    ##Full Model
    #print("\n\nFull Model unhealthy subjects")
    #print(classifier_dictionary[0][0])
    #classifier_dictionary[0][1].fit(x_full, y_full)
    preditedLabel_linear_full_unhealthy = pipe.predict(x_full_unhealthy)
    print("Intercept: ",pipe.named_steps['SVR_LINEAR'].intercept_)
    print("Coefficient: ",pipe.named_steps['SVR_LINEAR'].coef_)
    r2Score=r2_score(y_full_unhealthy,preditedLabel_linear_full_unhealthy)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(y_full_unhealthy, preditedLabel_linear_full_unhealthy)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    '''
    
    
    
    
    
    
    '''
    print("Result on Full Dataset:  ")
    preditedLabel_linear_full = classifier_dictionary[0][1].predict(x_data_val)
    r2Score=r2_score(y_data_val,preditedLabel_linear_full)
    print("Linear SVR R^2 Score:", r2Score)
    mse =mean_squared_error(y_data_val, preditedLabel_linear_full)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    #print("test result:", preditedLabel_linear)
    #print("test label:", predictedLabelFile)
    '''
    
    
    '''
    x_test_result = pd.DataFrame(data=x_test_full)
    x_test_result['brain_age'] = preditedLabel_linear
    x_test_result['age_diff'] = x_test_result['brain_age'] - x_test_result['age_at_cnb_x']
    x_test_result.to_csv('x_test_result.csv',encoding='utf-8',index=False,na_rep='NA')
    '''
    
    '''
    x_full_result = pd.DataFrame(data=x_full_feature)
    x_full_result['brain_age'] = preditedLabel_linear_full
    x_full_result['age_diff'] = x_full_result['brain_age'] - x_full_result['age_at_cnb_x']
    x_full_result.to_csv('x_full_result.csv',encoding='utf-8',index=False,na_rep='NA')
    '''
    
    '''
    x_full_unhealthy_result = pd.DataFrame(data=x_full_unhealthy_feature)
    x_full_unhealthy_result['brain_age'] = preditedLabel_linear_full_unhealthy
    x_full_unhealthy_result['age_diff'] = x_full_unhealthy_result['brain_age'] - x_full_unhealthy_result['age_at_cnb_x']
    x_full_unhealthy_result.to_csv('x_full_unhealthy_result.csv',encoding='utf-8',index=False,na_rep='NA')
    '''
    
    '''
    print("Prediction Result of Cross val:  ")
    y_pred_cross = cross_val_predict(make_pipeline(preprocessing.MinMaxScaler(), SVR(kernel='linear')), trainData, trainLabel, cv=kFold)
    mse =mean_squared_error(trainLabel, y_pred_cross)
    print("Linear SVR Mean Squared Error:",mse)
    rmse = math.sqrt(mse)
    print("Linear SVR Root Mean Squared Error:", rmse)
    x_train_result = pd.DataFrame(data=x_train_full)
    x_train_result['brain_age'] = y_pred_cross
    x_train_result['age_diff'] = x_train_result['brain_age'] - x_train_result['age_at_cnb_x']
    x_train_result.to_csv('x_train_result.csv',encoding='utf-8',index=False,na_rep='NA')
    '''
    
    '''
    x_test_result_Sensory_motor_Processing_Speed = pd.DataFrame(data=x_test_full)
    x_test_result_Sensory_motor_Processing_Speed['Predicted_MP_MP2RTCR'] = preditedLabel_linear
    x_test_result_Sensory_motor_Processing_Speed.to_csv('x_test_result_Sensory_motor_Processing_Speed.csv',encoding='utf-8',index=False,na_rep='NA')
    '''
    
    '''
    print("\n"+classifier_dictionary[1][0])
    classifier_dictionary[1][1].fit(trainData, trainLabel)
    preditedLabel_PLS = classifier_dictionary[1][1].predict(givenTestFeatures)
    r2Score_PLS=r2_score(predictedLabelFile,preditedLabel_PLS)
    print("PLS R^2 Score:", r2Score_PLS)
    mse_PLS =mean_squared_error(predictedLabelFile, preditedLabel_PLS)
    print("PLS Mean Squared Error:",mse_PLS)
    rmse_PLS = math.sqrt(mse_PLS)
    print("PLS Root Mean Squared Error:", rmse_PLS)
    '''
    
    
    '''
    plt.figure(figsize=(20,10))
    x_ax=range(160)
    plt.scatter(x_ax, predictedLabelFile, s=5, color="blue", label="original")
    plt.scatter(x_ax, preditedLabel_rbf, s=5, color="red", label="predicted")
    plt.show()
    
    plt.figure(figsize=(20,10))
    x_ax=range(160)
    plt.scatter(x_ax, predictedLabelFile, s=5, color="blue", label="original")
    plt.scatter(x_ax, preditedLabel_linear, s=5, color="red", label="predicted")
    plt.show()
    
    
    plt.figure(figsize=(20,10))
    x_ax=range(160)
    plt.scatter(x_ax, predictedLabelFile, s=5, color="blue", label="original")
    plt.scatter(x_ax, preditedLabel_PLS, s=5, color="red", label="predicted")
    plt.show()
    '''
    '''
    #sns.set(rc={'figure.figsize':(20,10)})
    plt.figure(figsize=(20,10))
    #plt.figure(figsize=(16, 9))
    data_test = {'ChronologicalAge': predictedLabelFile,'BrainAge':preditedLabel_linear}
    resultdf_f2 = pd.DataFrame(data_test)
    resultdf_f2.to_csv('F:/Pheno_data/dataSetFinalV/figure_male_female_model/resultdf_f2.csv', encoding='utf-8',na_rep='NA')

    #sns.regplot(x='ChronologicalAge',y='BrainAge', data=resultdf)
    sns.lmplot( x='ChronologicalAge',y='BrainAge', data=resultdf_f2, x_jitter=0.30)
    
    
    plt.figure(figsize=(20,10))
    #plt.figure(figsize=(16, 9))
    data_test_full = {'ChronologicalAge': y_full,'BrainAge':preditedLabel_linear_full}
    resultdf_f2_full = pd.DataFrame(data_test_full)
    resultdf_f2_full.to_csv('F:/Pheno_data/dataSetFinalV/figure_male_female_model/resultdf_f2_full.csv', encoding='utf-8',na_rep='NA')

    #sns.regplot(x='ChronologicalAge',y='BrainAge', data=resultdf)
    sns.lmplot( x='ChronologicalAge',y='BrainAge', data=resultdf_f2_full, x_jitter=0.50)
   
    


    plt.figure(figsize=(20,10))
    #plt.figure(figsize=(16, 9))
    data_test_unhealthy_full = {'ChronologicalAge': y_full_unhealthy,'BrainAge':preditedLabel_linear_full_unhealthy}
    resultdf_f2_unhealthy_full = pd.DataFrame(data_test_unhealthy_full)
    resultdf_f2_unhealthy_full.to_csv('F:/Pheno_data/dataSetFinalV/figure_male_female_model/resultdf_f2_unhealthy_full.csv', encoding='utf-8',na_rep='NA')

    #sns.regplot(x='ChronologicalAge',y='BrainAge', data=resultdf)
    sns.lmplot( x='ChronologicalAge',y='BrainAge', data=resultdf_f2_unhealthy_full, x_jitter=0.30)
    
    
    
    
    
    
    
    
    
    plt.figure(figsize=(20,10))
    fig, ax = plt.subplots()
    ax.scatter(predictedLabelFile, preditedLabel_linear,s=10)
    ax.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--', lw=2,marker='o',markersize=0.7)
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('Predicted Brain Age')
    plt.show()
    plt.savefig("F:/Pheno_data/dataSetFinalV/new_intersect_file/brain_age_fig/brain_age_test_sample_prediction.png")
    
    plt.figure(figsize=(20,10))
    fig, ax = plt.subplots()
    ax.scatter(y_full, preditedLabel_linear_full,s=5)
    ax.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--', lw=2)
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('Predicted Brain Age')
    plt.show()
    plt.savefig("F:/Pheno_data/dataSetFinalV/new_intersect_file/brain_age_fig/brain_age_full_sample_prediction.png")
    
    plt.figure(figsize=(20,10))
    fig, ax = plt.subplots()
    ax.scatter(y_full_unhealthy, preditedLabel_linear_full_unhealthy,s=10)
    ax.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--', lw=2,marker='o',markersize=0.7)
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('Predicted Brain Age')
    plt.show()
    plt.savefig("F:/Pheno_data/dataSetFinalV/new_intersect_file/brain_age_fig/brain_age_test_sample_prediction.png")

    plt.plot(y_full, preditedLabel_linear_full, 'o')
    #create scatter plot
    m, b = np.polyfit(y_full, preditedLabel_linear_full, 1)
    print("Slope", m)
    print("Intercept", b)
    #m = slope, b=intercept
    plt.plot(y_full, m*y_full + b)

    '''
    
    
    '''
    x_full_r = pd.read_csv("F:/fmri_data_test/nextPhase/RFE/x_full_result.csv")
    
    x_patients = pd.read_csv("F:/fmri_data_test/nextPhase/RFE/x_full_unhealthy_result.csv")
    '''
    
    
    
    '''
    print("Slope & Intercept for patients: ")
    fig, px = plt.subplots()
    px.plot(y_full_unhealthy, preditedLabel_linear_full_unhealthy, 'o')
    #create scatter plot
    slopeUnhealthy, intercptUnhealthy = np.polyfit(y_full_unhealthy, preditedLabel_linear_full_unhealthy, 1)
    print("slopeUnhealthy", slopeUnhealthy)
    print("intercptUnhealthy", intercptUnhealthy)
    #m = slope, b=intercept
    px.plot(y_full_unhealthy, slopeUnhealthy*y_full_unhealthy + intercptUnhealthy)
    plt.show()
    '''
    
    
    '''
    fig, pxAgediff = plt.subplots()
    pxAgediff.plot(y_full_unhealthy, x_patients['age_diff'], 'o')
    #create scatter plot
    mUn, bUn = np.polyfit(y_full_unhealthy, x_patients['age_diff'], 1)
    print("slopeUnhealthyAgediff", mUn)
    print("InterceptUnhealthyAgediff", bUn)
    #m = slope, b=intercept
    pxAgediff.plot(y_full_unhealthy, mUn*y_full_unhealthy + bUn)
    plt.show()
    '''
    

    
    '''
    brainAge_vs_trueAge = pd.DataFrame()
    brainAge_vs_trueAge["Brain Age"]=x_full_r["brain_age"]
    brainAge_vs_trueAge["Chronological Age"]=x_full_r["age_at_cnb_x"]
    sns.lmplot( x='Chronological Age',y='Brain Age', data=brainAge_vs_trueAge, x_jitter=0.50)
    
    fig, mx = plt.subplots()
    mx.plot(y_full, preditedLabel_linear_full, 'o')
    #create scatter plot
    slope, intercpt = np.polyfit(y_full, preditedLabel_linear_full, 1)
    print("Slope", slope)
    print("Intercept", intercpt)
    #m = slope, b=intercept
    mx.plot(y_full, slope*y_full + intercpt)
    plt.show()
    
    x_full_r["Cole_bias_free_brain_age"] =  (x_full_r["brain_age"]-intercpt)/slope
    x_full_r["Cole_bias_adjusted_age_diff"] = x_full_r["Cole_bias_free_brain_age"] - x_full_r['age_at_cnb_x']
    
    
    
    age_diff_vs_trueAge = pd.DataFrame()
    age_diff_vs_trueAge["Age_difference"]=x_full_r["age_diff"]
    age_diff_vs_trueAge["Chronological Age"]=x_full_r["age_at_cnb_x"]
    
    sns.lmplot( x='Chronological Age',y='Age_difference', data=age_diff_vs_trueAge, x_jitter=0.50)
    
    
    fig, ax = plt.subplots()
    ax.plot(y_full, x_full_r['age_diff'], 'o')
    #create scatter plot
    m, b = np.polyfit(y_full, x_full_r['age_diff'], 1)
    print("Slope", m)
    print("Intercept", b)
    #m = slope, b=intercept
    ax.plot(y_full, m*y_full + b)
    plt.show()
    
    x_full_r["offset"] = m * x_full_r["age_at_cnb_x"] + b
    x_full_r["bias_free_brain_age"] = x_full_r["brain_age"]-x_full_r["offset"]
    x_full_r["bias_adjusted_age_diff"] = x_full_r["bias_free_brain_age"] - x_full_r['age_at_cnb_x']
    
    
    x_full_r.to_csv('F:/fmri_data_test/nextPhase/RFE/x_full_result_bias_free.csv', encoding='utf-8',na_rep='NA')
    
    bias_adjusted_age_diff_vs_trueAge = pd.DataFrame()
    bias_adjusted_age_diff_vs_trueAge["Age_difference"]=x_full_r["bias_adjusted_age_diff"]
    bias_adjusted_age_diff_vs_trueAge["Chronological Age"]=x_full_r["age_at_cnb_x"]
    
    sns.lmplot( x='Chronological Age',y='Age_difference', data=bias_adjusted_age_diff_vs_trueAge, x_jitter=0.50)
    
    fig, bx = plt.subplots()
    bx.plot(y_full, x_full_r["bias_adjusted_age_diff"], 'o')
    #create scatter plot
    m, b = np.polyfit(y_full, x_full_r["bias_adjusted_age_diff"], 1)
    print("Slope", m)
    print("Intercept", b)
    #m = slope, b=intercept
    bx.plot(y_full, m*y_full + b)
    plt.show()
    
    Cole_bias_adjusted_age_diff_vs_trueAge = pd.DataFrame()
    Cole_bias_adjusted_age_diff_vs_trueAge["Age_difference"]=x_full_r["Cole_bias_adjusted_age_diff"]
    Cole_bias_adjusted_age_diff_vs_trueAge["Chronological Age"]=x_full_r["age_at_cnb_x"]

    sns.lmplot( x='Chronological Age',y='Age_difference', data=Cole_bias_adjusted_age_diff_vs_trueAge, x_jitter=0.50)
    
    
    fig, nx = plt.subplots()
    nx.plot(y_full, x_full_r["Cole_bias_adjusted_age_diff"], 'o')
    #create scatter plot
    m, b = np.polyfit(y_full, x_full_r["Cole_bias_adjusted_age_diff"], 1)
    print("Slope", m)
    print("Intercept", b)
    #m = slope, b=intercept
    nx.plot(y_full, m*y_full + b)
    plt.show()
    
    bias_adjusted_age_vs_trueAge = pd.DataFrame()
    bias_adjusted_age_vs_trueAge["Bias adjusted Brain age"]=x_full_r["bias_free_brain_age"]
    bias_adjusted_age_vs_trueAge["Chronological Age"]=x_full_r["age_at_cnb_x"]
    sns.lmplot( x='Chronological Age',y='Bias adjusted Brain age', data=bias_adjusted_age_vs_trueAge, x_jitter=0.50)
    
    x_full_r.hist(column='bias_adjusted_age_diff')
    
    x_full_r.hist(column='Cole_bias_adjusted_age_diff')
    
    
    x_patients["Cole_bias_free_brain_age"] =  (x_patients["brain_age"]-intercpt)/slope
    x_patients["Cole_bias_adjusted_age_diff"] = x_patients["Cole_bias_free_brain_age"] - x_patients['age_at_cnb_x']
    
    Cole_bias_adjusted_age_diff_vs_trueAge_unhealthy = pd.DataFrame()
    Cole_bias_adjusted_age_diff_vs_trueAge_unhealthy["Age_difference"]=x_patients["Cole_bias_adjusted_age_diff"]
    Cole_bias_adjusted_age_diff_vs_trueAge_unhealthy["Chronological Age"]=x_patients["age_at_cnb_x"]
    sns.lmplot( x='Chronological Age',y='Age_difference', data=Cole_bias_adjusted_age_diff_vs_trueAge_unhealthy, x_jitter=0.50)
    
    x_patients["offset"] = m * x_patients["age_at_cnb_x"] + b
    x_patients["bias_free_brain_age"] = x_patients["brain_age"]-x_patients["offset"]
    x_patients["bias_adjusted_age_diff"] = x_patients["bias_free_brain_age"] - x_patients['age_at_cnb_x']
    x_patients.to_csv('F:/fmri_data_test/nextPhase/RFE/x_patients_bias_free.csv', encoding='utf-8',na_rep='NA')
    
    bias_adjusted_age_diff_vs_trueAge_unhealthy = pd.DataFrame()
    bias_adjusted_age_diff_vs_trueAge_unhealthy["Age_difference"]=x_patients["bias_adjusted_age_diff"]
    bias_adjusted_age_diff_vs_trueAge_unhealthy["Chronological Age"]=x_patients["age_at_cnb_x"]
    sns.lmplot( x='Chronological Age',y='Age_difference', data=bias_adjusted_age_diff_vs_trueAge_unhealthy, x_jitter=0.50)
  
    
    x_patients.hist(column='bias_adjusted_age_diff')
    x_patients.hist(column='Cole_bias_adjusted_age_diff')
    
    #print("length: ", len(x_patients["bias_adjusted_age_diff"]))
    fig, unx1 = plt.subplots()
    unx1.hist(x_patients["bias_adjusted_age_diff"], normed=True)
    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(x_patients["bias_adjusted_age_diff"]))
    # lets try the normal distribution first
    m, s = stats.norm.fit(x_patients["bias_adjusted_age_diff"]) # get mean and standard deviation  
    #print("Mean :", m)
    #print("sd: ", s)
    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
    unx1.plot(lnspc, pdf_g, label="Norm") # plot it
    plt.show()
    
    fig, unx2 = plt.subplots()
    unx2.hist(x_patients["Cole_bias_adjusted_age_diff"], normed=True)
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(x_patients["Cole_bias_adjusted_age_diff"]))
    # lets try the normal distribution first
    m, s = stats.norm.fit(x_patients["Cole_bias_adjusted_age_diff"]) # get mean and standard deviation  
    print("Mean :", m)
    print("sd: ", s)
    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
    unx2.plot(lnspc, pdf_g, label="Norm") # plot it
    plt.show()
    '''
    
    '''
    m, b = best_fit_slope_and_intercept(y_full,x_full_r['age_diff'])

    print(m,b)
    regression_line = []
    for x in y_full:
        regression_line.append((m*x)+b)
        
    plt.scatter(y_full,x_full_r['age_diff'],color='#003F72')
    plt.plot(y_full, regression_line)
    plt.show()
    '''
    
    
    '''
    m, b = best_fit_slope_and_intercept(y_full,preditedLabel_linear_full)

    print(m,b)
    regression_line = []
    for x in y_full:
        regression_line.append((m*x)+b)
        
    plt.scatter(y_full,preditedLabel_linear_full,color='#003F72')
    plt.plot(y_full, regression_line)
    plt.show()
    '''
    
    #sns.stripplot(y_full, preditedLabel_linear_full, jitter=0.2, size=2)
    
    
    
    
    
    #return mostAccurateClassifier
    
    


################ Initialize Classification ###############
'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/Final/shen/Shen_Roi_dataSetMedSIP_AGE.csv", header=None, skiprows=1)
X_data = data.iloc[:,0:265]
Y_data = data.iloc[:,265]
'''


'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/Final/ICA_dataSetMedSIP_AGE.csv", header=None, skiprows=1)
X_data = data.iloc[:,0:100]
Y_data = data.iloc[:,100]
'''

'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/Final/AAL_116_Roi_dataSetMedSIP_AGE.csv", header=None, skiprows=1)
X_data = data.iloc[:,0:116]
Y_data = data.iloc[:,116]
'''

'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/freeSrfrCleanData.csv", header=None,skiprows=1)
X_data = data.iloc[:,0:152]
Y_data = data.iloc[:,152]
'''


'''
data = pd.read_csv("intersect_ICA_AAL_freeSrfr2.csv")
X_data = data.iloc[:,1:369]
Y_data = data.iloc[:,0]
'''

'''
rfecv_featureCoeff_ = pd.read_csv("rfecv_featureCoeff.csv")
#rfecv_rmse_featureCoeff = pd.read_csv("/data/mialab/users/bray14/RFE/rfecv_rmse_featureCoeff.csv")
#rfecv_featureCoeff = pd.read_csv("/data/mialab/users/bray14/RFE/pca_rfe.csv")
data = pd.read_csv("intersect_ICA_AAL_freeSrfr2.csv")
#data = pd.read_csv("/data/mialab/users/bray14/RFE/pca.csv")
#data = data.drop('Unnamed: 0', axis=1)
X_data = data[rfecv_featureCoeff_.attr]
# = data.iloc[:,0]
Y_data = pd.read_csv("Y_dataset.csv")
Y_data = Y_data['age_at_cnb']
'''



'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr2_reduced.csv", header=None,skiprows=1)
X_data = data.iloc[:,1:222]
Y_data = data.iloc[:,0]
'''

'''
data1 = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_fnc_l_2.csv")
X_data1 = data1.iloc[:,4:1544]
Y_data1 = data1['age_at_cnb']
'''

'''
data1 = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_fnc_ica_2_filtered.csv")
X_data1 = data1.iloc[:,5:105]
Y_data1 = data1['age_at_cnb']
'''


'''
rfecv_featureCoeff_ = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/temp_rfe/server/rfecv_rmse_featureCoeff.csv")
data1 = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_fnc_ica_2_filtered.csv")
X_data1 = data1.iloc[:,5:105]
X_data1 = X_data1[rfecv_featureCoeff_.attr]
Y_data1 = data1['age_at_cnb']
'''


'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fnc_l_2.csv")
data.drop(["age_at_cnb_y","Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
X_data = data.iloc[:,5:1913]
Y_data = data['age_at_cnb_x']
'''


'''
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fnc_ica_filtered.csv")
data.drop(["age_at_cnb_y",'Sex_y',"Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
X_data = data.iloc[:,5:473]
Y_data = data['age_at_cnb_x']
'''


rfecv_featureCoeff_all = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/temp_rfe/server/rfecv_rmse_featureCoeff_all.csv")
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fnc_ica_filtered.csv")
data.drop(["age_at_cnb_y",'Sex_y',"Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
data['Sex_coded'] = data['Sex_x'].map( {'M':0, 'F':1} )
X_data = data.iloc[:,5:474]
X_data = X_data[rfecv_featureCoeff_all.attr]
Y_data = data['age_at_cnb_x']


'''
count_age =data.groupby('age_at_cnb_x').Sex_x.count().values
print(count_age)

plt.figure(figsize=(20,10))
age_labels = ['8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20','21']
plot_data = data.groupby('age_at_cnb_x').Sex_x.count().values
ax = plt.subplot()
male_plt_position = np.array(range(len(age_labels)))
tick_spacing = np.array(range(len(age_labels)))
ax.bar(male_plt_position, plot_data,color='b')
plt.xticks(tick_spacing,  age_labels)
ax.set_ylabel("Frequency")
ax.set_xlabel("Age")
ax.set_title("Frequency of subjects by age",fontsize=14)
plt.legend(loc='best')
plt.show()
'''

'''
count =data.groupby('age_at_cnb_x').Sex_x.value_counts() 
print(count)

plt.figure(figsize=(20,10))
age_labels = ['8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20','21']
male_data = data[data.Sex_x == "M"].groupby('age_at_cnb_x').Sex_x.count().values
print(male_data)
female_data = data[data.Sex_x == "F"].groupby('age_at_cnb_x').Sex_x.count().values
print(female_data)
ax = plt.subplot()
male_plt_position = np.array(range(len(age_labels)))
female_plt_position = np.array(range(len(age_labels)))+0.4
tick_spacing = np.array(range(len(age_labels)))+0.4
ax.bar(male_plt_position, male_data,width=0.4,label='Male',color='b')
ax.bar(female_plt_position, female_data,width=0.4,label='Female',color='r')
plt.xticks(tick_spacing,  age_labels)
ax.set_ylabel("Frequency")
ax.set_xlabel("Age")
ax.set_title("Frequency of subjects by age/ Gender",fontsize=14)
plt.legend(loc='best')
plt.show()
'''








'''
#data.groupby('Sex_coded').age_at_cnb_x.plot(kind='kde')
#sns.distplot(data['age_at_cnb_x'], fit=norm, kde=False)
#data=data[data['Sex_x'].str.match('M')]
#data=data[data['Sex_x'].str.match('F')]
#data=data[data['Sex_x']=='M']
#data['Sex_x'].str.replace([0,1],['M','F'],inplace=True)
#data=data[data['Sex_x'].str.replace([0,1],['M','F'], regex=False)]
#data=data['Sex_x'].replace([0,1],['M','F'], inplace=True)
#data=data['Sex_x'].str.replace([0,1],['M','F'])
X_data = data.iloc[:,5:474]
X_data = X_data[rfecv_featureCoeff_all.attr]
X_data['Sex_coded'] = data['Sex_coded']
Y_data = data['age_at_cnb_x']

'''



'''
rfecv_featureCoeff_all = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/temp_rfe/server/rfecv_rmse_featureCoeff_all.csv")
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fncICA_control.csv")
data_unhealthy = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fncICA_unhealthy.csv")



data['Sex_coded'] = data['Sex_x'].map( {'M':0, 'F':1} )
data=data[data.WRAT_CR_RAW_x.notnull()]
data=data[data.WRAT_CR_STD_x.notnull()]
data=data[data.PCPT_T_TP.notnull()]
data=data[data.MP_MP2RTCR.notnull()]
data=data[data.LNB_MCR.notnull()]
#X_data = data.iloc[:,5:474]
X_data = data
X_data = X_data[rfecv_featureCoeff_all.attr]
X_data['Sex_coded'] = data['Sex_coded']
X_data['SUBJID'] = data['SUBJID']
X_data['age_at_cnb_x'] = data['age_at_cnb_x']
X_data['Sex_x'] = data['Sex_x']
X_data['Med_Rating_x'] = data['Med_Rating_x']
X_data['SIP_COUNT'] = data['SIP_COUNT']
X_data['WRAT_CR_RAW_x'] = data['WRAT_CR_RAW_x']
X_data['WRAT_CR_STD_x'] = data['WRAT_CR_STD_x']
X_data['PCPT_T_TP'] = data['PCPT_T_TP']
X_data['MP_MP2RTCR'] = data['MP_MP2RTCR']
X_data['LNB_MCR'] = data['LNB_MCR']
Y_data = data['age_at_cnb_x']
X_data_val = X_data.iloc[:,0:189]





data_unhealthy['Sex_coded'] = data_unhealthy['Sex_x'].map( {'M':0, 'F':1} )
data_unhealthy=data_unhealthy[data_unhealthy.WRAT_CR_RAW_x.notnull()]
data_unhealthy=data_unhealthy[data_unhealthy.WRAT_CR_STD_x.notnull()]
data_unhealthy=data_unhealthy[data_unhealthy.PCPT_T_TP.notnull()]
data_unhealthy=data_unhealthy[data_unhealthy.MP_MP2RTCR.notnull()]
data_unhealthy=data_unhealthy[data_unhealthy.LNB_MCR.notnull()]
#X_data = data.iloc[:,5:474]
X_data_unhealthy = data_unhealthy
X_data_unhealthy = X_data_unhealthy[rfecv_featureCoeff_all.attr]
X_data_unhealthy['Sex_coded'] = data_unhealthy['Sex_coded']
X_data_unhealthy['SUBJID'] = data_unhealthy['SUBJID']
X_data_unhealthy['age_at_cnb_x'] = data_unhealthy['age_at_cnb_x']
X_data_unhealthy['Sex_x'] = data_unhealthy['Sex_x']
X_data_unhealthy['Med_Rating_x'] = data_unhealthy['Med_Rating_x']
X_data_unhealthy['SIP_COUNT'] = data_unhealthy['SIP_COUNT']
X_data_unhealthy['WRAT_CR_RAW_x'] = data_unhealthy['WRAT_CR_RAW_x']
X_data_unhealthy['WRAT_CR_STD_x'] = data_unhealthy['WRAT_CR_STD_x']
X_data_unhealthy['PCPT_T_TP'] = data_unhealthy['PCPT_T_TP']
X_data_unhealthy['MP_MP2RTCR'] = data_unhealthy['MP_MP2RTCR']
X_data_unhealthy['LNB_MCR'] = data_unhealthy['LNB_MCR']
Y_data_unhealthy = data_unhealthy['age_at_cnb_x']
X_data_unhealthy_val = X_data_unhealthy.iloc[:,0:189]

'''

'''
data1 = pd.read_csv("F:/fmri_data_test/nextPhase/RFE/x_test_result.csv")
data1['Sex_coded'] = data1['Sex_x'].map( {'M':0, 'F':1} )
X_data_new = pd.DataFrame()
X_data_new['Sex_coded'] = data1['Sex_coded'].astype(float)
X_data_new['age_at_cnb_x'] = data1['age_at_cnb_x'].astype(float)
X_data_new['age_diff'] = data1['age_diff'].astype(float)
X_data_new['WRAT_CR_RAW_x'] = data1['WRAT_CR_RAW_x'].astype(float)
X_data_new['WRAT_CR_STD_x'] = data1['WRAT_CR_STD_x'].astype(float)
X_data_new['PCPT_T_TP'] = data1['PCPT_T_TP'].astype(float)
X_data_new['MP_MP2RTCR'] = data1['MP_MP2RTCR'].astype(float)
X_data_new['LNB_MCR'] = data1['LNB_MCR'].astype(float)
X_data_new['Sex_x'] = data1['Sex_x']
Y_data_new = data1['MP_MP2RTCR'].round(0).astype(int)
'''

'''

#Male

rfecv_featureCoeff_all = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/temp_rfe/server/rfecv_rmse_featureCoeff_all.csv")
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fnc_ica_filtered.csv")
data.drop(["age_at_cnb_y",'Sex_y',"Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
dataMale=data[data['Sex_x'].str.match('M')]
dataFemale=data[data['Sex_x'].str.match('F')]


X_train_male = dataMale.iloc[:,5:473]
X_train_male = X_train_male[rfecv_featureCoeff_all.attr]
y_train_male = dataMale['age_at_cnb_x']
'''
'''
X_train = dataMale.iloc[0:400,5:473]
X_train = X_train[rfecv_featureCoeff_all.attr]
y_train = dataMale['age_at_cnb_x']
y_train = y_train.iloc[0:400]
'''
'''
X_test_male = dataMale.iloc[:,5:473]
X_test_male = X_test_male[rfecv_featureCoeff_all.attr]
y_test_male = dataMale['age_at_cnb_x']
#y_test = y_test.iloc[400:444]


X_test_female_MM = dataFemale.iloc[:,5:473]
X_test_female_MM = X_test_female_MM[rfecv_featureCoeff_all.attr]
y_test_female_MM = dataFemale['age_at_cnb_x']


X_train_female = dataFemale.iloc[:,5:473]
X_train_female = X_train_female[rfecv_featureCoeff_all.attr]
y_train_female = dataFemale['age_at_cnb_x']

X_test_female = dataFemale.iloc[:,5:473]
X_test_female = X_test_female[rfecv_featureCoeff_all.attr]
y_test_female = dataFemale['age_at_cnb_x']

X_test_male_FM = dataMale.iloc[:,5:473]
X_test_male_FM = X_test_male_FM[rfecv_featureCoeff_all.attr]
y_test_male_FM = dataMale['age_at_cnb_x']
'''

#fEMALE
'''
rfecv_featureCoeff_all = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/temp_rfe/server/rfecv_rmse_featureCoeff_all.csv")
data = pd.read_csv("F:/Pheno_data/dataSetFinalV/new_intersect_file/intersect_ICA_AAL_freeSrfr_fnc_ica_filtered.csv")
data.drop(["age_at_cnb_y",'Sex_y',"Med_Rating_y","SIP_COUNT_y"],axis=1,inplace=True)
dataMale=data[data['Sex_x'].str.match('M')]
dataFemale=data[data['Sex_x'].str.match('F')]


X_train = dataFemale.iloc[:,5:473]
X_train = X_train[rfecv_featureCoeff_all.attr]
y_train = dataFemale['age_at_cnb_x']
'''

'''
X_train = dataFemale.iloc[0:491,5:473]
X_train = X_train[rfecv_featureCoeff_all.attr]
y_train = dataFemale['age_at_cnb_x']
y_train = y_train.iloc[0:491]
'''


'''
X_test = dataMale.iloc[:,5:473]
X_test = X_test[rfecv_featureCoeff_all.attr]
y_test = dataMale['age_at_cnb_x']
'''

'''
X_test = dataFemale.iloc[491:545,5:473]
X_test = X_test[rfecv_featureCoeff_all.attr]
y_test = dataFemale['age_at_cnb_x']
y_test = y_test.iloc[491:545]

'''





'''
ICA_COMP = rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('ICA_COMP_')]
AAL_ROI = rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('AAL_ROI_')]
Cortical_Volume_Thickness = rfecv_featureCoeff_all[rfecv_featureCoeff_all["attr"].str.match('FNC_ICA_|ICA_COMP_|AAL_ROI_')==False]
FNC_ICA = rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('FNC_ICA_')]
'''

'''
print (rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('FNC_ICA_')])
print (rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('ICA_COMP_')])
print (rfecv_featureCoeff_all[rfecv_featureCoeff_all['attr'].str.match('AAL_ROI_')])
print(rfecv_featureCoeff_all[rfecv_featureCoeff_all["attr"].str.match('FNC_ICA_|ICA_COMP_|AAL_ROI_')==False])
'''


X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.10)

#X_train, X_test, y_train, y_test = train_test_split(X_data1, Y_data1, test_size = 0.10)
#X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.30)
#X_train_data = X_train.iloc[:,0:189]
#X_test_data = X_test.iloc[:,0:189]

#X_train, X_test, y_train, y_test = train_test_split(X_data_new, Y_data_new, test_size = 0.10)
#X_train_data_new = X_train.iloc[:,0:3]
#X_test_data_new = X_test.iloc[:,0:3]

trainModelWithDataset(X_train, y_train, X_test, y_test)
#trainModelWithDataset(X_train_data, y_train, X_test_data, y_test, X_train,X_test,X_data_val,Y_data,X_data)

#trainModelWithDataset(X_train, y_train, X_test, y_test,X_data_val,Y_data,X_data,X_data_unhealthy_val,Y_data_unhealthy,X_data_unhealthy)

#trainModelWithDataset(X_train_male, y_train_male, X_test_male, y_test_male,X_data_val,Y_data,X_data,X_data_unhealthy_val,Y_data_unhealthy,X_data_unhealthy,X_test_female_MM,y_test_female_MM)

#trainModelWithDataset(X_train_female, y_train_female, X_test_female, y_test_female,X_data_val,Y_data,X_data,X_data_unhealthy_val,Y_data_unhealthy,X_data_unhealthy,X_test_male_FM,y_test_male_FM)

'''
print (pd.to_numeric(X_train_data_new['Sex_coded'], errors='coerce').isnull())
X_train_data_new[~X_train_data_new.applymap(np.isreal).all(1)]
X_train_data_new.applymap(np.isreal)
X_test_data_new[~X_test_data_new.applymap(np.isreal).all(1)]
X_test_data_new.applymap(np.isreal)
'''

#y_train[~y_train.applymap(np.isreal).all(1)]

#num_df = (df.drop(data_columns, axis=1).join(df[data_columns].apply(pd.to_numeric, errors='coerce')))


#np.where(np.any(np.isnan(X_train_data_new.convert_objects(convert_numeric=True)), axis=1))