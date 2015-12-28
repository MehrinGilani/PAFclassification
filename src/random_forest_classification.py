## this file reads array of all features and does classification_functions
import matplotlib;
import math;
import matplotlib.pyplot as plt;
import numpy as np;
import wfdb,sys,re;
import scipy;
import scipy.sparse
import scipy.stats
import numpy.linalg as LA;
import pywt;
import os;


from scipy import signal;
from _wfdb import calopen, aduphys;
from wfdb import WFDB_Siginfo
from matplotlib.lines import lineStyles
from _wfdb import strtim
from matplotlib.pyplot import show
from numpy import dtype, argmin, shape
from scipy.stats.mstats_basic import kurtosistest
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing;
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn import cross_validation
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.decomposition import PCA
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier
import wfdb_setup as ws;
import process_rr as pr;
import data_cleaning as dc;
import graphs 
import read_write as rw;
import non_linear_measures as nlm;
import classification_functions as cl
import sklearn

##############################################################################

output_folder="/home/ubuntu/Documents/Thesis_work/testing/rf_sklearn/"
#output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/afpdb_test_records/"

# ### read values from text files ######
# all_features=rw.read_features_frm_file(output_folder,"all_features_pickle.txt")
# rw.write_value(all_features,output_folder,"list_of_list_before_cleaning","w")
# global_vocab=rw.read_features_frm_file(output_folder,"global_vocab_pickle.txt")
# rec_name_array=rw.read_features_frm_file(output_folder,"rec_name_array_pickle.txt")
# 
# 
# #selected_feature_index=rw.read_features_frm_file(output_folder,"rfecv_selected13_features.txt")
# ##################### change key value pairs of global vocab ####################
# inv_global_vocab = dict(zip(global_vocab.values(), global_vocab.keys()))
# #print type(inv_global_vocab.values())
# all_features_list=inv_global_vocab.values()
# np.savetxt(output_folder+"all_features_list.txt",all_features_list,fmt="%s",delimiter=',',newline='\n')

feature_matrix = np.loadtxt('/home/ubuntu/Documents/Thesis_work/testing/rf_sklearn/SQ_filtered_Xtrain_pwave_29fs_no_label.csv', delimiter=",")
print ("type of feature matrix is : " + str(type(feature_matrix)))

#feature_matrix=feature_matrix[:,0:1]

y=[]
#generate class labels
f=open('/home/ubuntu/Documents/Thesis_work/testing/rf_sklearn/labels.csv', 'r')
for line in f:
    if 'normal' in line:
        y.append(int(0))
    elif 'patient' in line:
        y.append(int(1))

y_arr=np.asarray(y)
#print ("label nd array is: " + str((y_arr)))


#convert list of lists to matrix
#all_feature_matrix=cl.covert_array_to_matrix(all_features,len(all_features));

#print all_feature_matrix
#print ("shape of all feature matrix  is: " + str(all_feature_matrix.shape))
#################### SEPARATING EVALUATION DATA #########################
#X_cv, X_eval, y_cv, y_eval = cross_validation.train_test_split(all_feature_matrix, y, test_size=0.2, random_state=0)



############## with normalisation ######################
# Classification
# normalised="  "
# normalized_matrix=cl.normalise_mean_var(all_feature_matrix)
# rw.write_df_to_csv(normalized_matrix, csv_header, output_folder, "features_normalised_test.csv")
# exit()


#print ("type of normalised_matrix is: " + str(type(normalized_matrix)))
#print("normalized matrix is: ")
#print normalized_matrix
#rw.write_value(normalized_matrix,output_folder,"all_features_normalized.txt",'w')

#feature_matrix=normalized_matrix;
#feature_matrix=all_feature_matrix
method='new feature'
###############################################################################
if method == 'new feature':
    print(" new feature")
# Classification and ROC analysis
 
    # Run classifier with cross-validation and plot ROC curves
     
    folds=len(y_arr)
    
    cv = StratifiedKFold(y, n_folds=folds)
    classifier_1 = RandomForestClassifier(n_estimators=100)
    classifier_2 = RandomForestClassifier(n_estimators=100)
    
###################################################
### implementing 2 classifiers

    y_test_report=[];
    y_predicted_report=[]
      
      
    all_indexes=[]
    index_list=[]
    for i, (train, test) in enumerate(cv):
        proba0_train=[]
        proba1_train=[]
        proba0_test=[]
        proba1_test=[]
        pwave_pred_normal_index=[]
        pwave_pred_patient_index=[]
         
        rr_pred_normal_index=[]
        rr_pred_patient_index=[]
        y_predicted=[-1]*(len(y_arr[test]))
         
        
        y_predicted_pwave=[]
        y_predicted_rr=[]
         
        feature_pwave=feature_matrix[:,0:2]
        feature_rr=feature_matrix[:,2:-1]
        
        local_index=[]
        local_patient_index=[]
        local_normal_index=[]
        
        classifier_1.fit(feature_rr[train], y_arr[train]) 
        l_feature_pred_prob_train=(classifier_1.predict_proba(feature_rr[train]))
        #print("l_feature_pred_prob_train is : " + str(feature_pred_prob))
        
        ## make 2 new features for train using predic proba ####
        for i in range(len(l_feature_pred_prob_train)):
            proba0_train.append(l_feature_pred_prob_train[i,0])
            proba1_train.append(l_feature_pred_prob_train[i,1])
        
        #print (" proba1_train is: "  + str(proba1_train))
       
        y_predicted_rr=(classifier_1.predict(feature_rr[test]))
        l_feature_pred_prob_test=(classifier_1.predict_proba(feature_rr[test]))
        ## make 2 new features for test using predic proba ####
        for i in range(len(l_feature_pred_prob_test)):
            proba0_test.append(l_feature_pred_prob_test[i,0])
            proba1_test.append(l_feature_pred_prob_test[i,1])
        
#         for i in range(len(y_predicted_rr)):
#             if y_predicted_rr[i] != y_arr[test[i]]:
#                 print ("predicted y is: " + str(y_predicted_rr[i]) + " predicted probability is: " + str(feature_pred_prob_train[i]))
            
            #y_predicted_rr=[1]*(len(y_arr[test]))
        
        
        
        for i in range(len(y_predicted_rr)):   
        #for predicted_class in y_predicted_pwave:
            if y_predicted_rr[i] == 0:
                local_index.append(i)
                rr_pred_normal_index.append(test[i])
            elif y_predicted_rr[i]== 1:
                rr_pred_patient_index.append(test[i])
                local_patient_index.append(i)
                
  
             
        
        ## make new feature train matrix here: 
        added_feature_matrix_train=np.zeros((len(train),((shape(feature_pwave)[1])+2)),dtype=float)
        added_feature_matrix_train[:,:-2]=feature_pwave[train]
        added_feature_matrix_train[:,-2]=proba0_train
        added_feature_matrix_train[:,-1]=proba1_train
        #print ("shape of added feature matrix is: " + str(shape(added_feature_matrix_train)))
        
        ## make new feature test matrix here: 
#         added_feature_matrix_test=np.zeros((len(test),((shape(feature_pwave)[1])+2)),dtype=float)
#         added_feature_matrix_test[:,:-2]=feature_pwave[test]
#         added_feature_matrix_test[:,-2]=proba0_test
#         added_feature_matrix_test[:,-1]=proba1_test
#         print ("shape of added feature matrix is: " + str(shape(added_feature_matrix_test)))
        
        classifier_2.fit(added_feature_matrix_train,y_arr[train])
        
        pwave_y_test=added_feature_matrix_train[rr_pred_patient_index]
        #print ("rr_y_test is:  " + str(rr_y_test))
        y_predicted_pwave=classifier_2.predict(pwave_y_test)
         
        
        for i in range(len(y_predicted_pwave)):
            if y_predicted_pwave[i] == 0:
                pwave_pred_normal_index.append(local_patient_index[i])
            elif y_predicted_pwave[i] ==1:
                pwave_pred_patient_index.append(local_patient_index[i])
                 
        
       
        
    #     for val in rr_pred_normal_index:
    #         if y_predicted_pwave[val] == y_predicted_rr[val]:
    #             y_predicted[val]=0;
    #         else:
    #             y_predicted[val]=1
        for val in local_index:
            if y_predicted[val] != -1:
                print("error in local _index loop")
                #exit()
            
            y_predicted[val]=0;
        
        for val in pwave_pred_normal_index:
            if y_predicted[val] != -1:
                print("error in pwave_pred_normal loop")
                #exit()
            y_predicted[val]=0;
       
        for val in pwave_pred_patient_index:
            if y_predicted[val] != -1:
                print('val is : ' + str(val))
                print('y predicted[val] is : ' + str(y_predicted[val]))
                print("error in pwave_pred_patient loop")
                #exit()
            #if y_predicted_pwave[val] != y_predicted_rr[val]:
            y_predicted[val]=1;
            
        
        #y_predicted=y_predicted_rr
        
        
        ##make predicted array
        #print y_predicted
         
        y_predicted_report.extend(y_predicted)
        y_test_report.extend(y_arr[test])
         
     
     
    print y_test_report
    print y_predicted_report
    ####### Compute confusion matrix #######
    cm = metrics.confusion_matrix(y_test_report, y_predicted_report)
    np.set_printoptions(precision=2)
    print('Confusion matrix for: rf_sklearn')
    print(cm)
     
       
    print ("overall accuracy score of the classifier is")
    print metrics.accuracy_score(y_test_report, y_predicted_report)
    target_names = ['class 0', 'class 1']
    print(classification_report(np.array(y_test_report), np.array(y_predicted_report), target_names=target_names));
###############################################################################
elif method == ' no new feature':
    print("in no new feature")
# Classification and ROC analysis
 
    # Run classifier with cross-validation and plot ROC curves
     
    folds=len(y_arr)
    
    cv = StratifiedKFold(y, n_folds=folds)
    classifier_1 = RandomForestClassifier(n_estimators=100)
    classifier_2 = RandomForestClassifier(n_estimators=100)
   
###################################################

### implementing 2 classifiers

    y_test_report=[];
    y_predicted_report=[]
      
      
    all_indexes=[]
    index_list=[]
    for i, (train, test) in enumerate(cv):
      
        pwave_pred_normal_index=[]
        pwave_pred_patient_index=[]
         
        rr_pred_normal_index=[]
        rr_pred_patient_index=[]
        y_predicted=[-1]*(len(y_arr[test]))
         
        
        y_predicted_pwave=[]
        y_predicted_rr=[]
         
        feature_pwave=feature_matrix[:,0:1]
        feature_rr=feature_matrix[:,2:-1]
        
        local_index=[]
        local_patient_index=[]
        local_normal_index=[]
        
        classifier_1.fit(feature_rr[train], y_arr[train]) 
    
        y_predicted_rr=(classifier_1.predict(feature_rr[test]))
        
        
       
            
            #y_predicted_rr=[1]*(len(y_arr[test]))
        
        
        
        for i in range(len(y_predicted_rr)):   
        #for predicted_class in y_predicted_pwave:
            if y_predicted_rr[i] == 0:
                local_index.append(i)
                rr_pred_normal_index.append(test[i])
            elif y_predicted_rr[i]== 1:
                rr_pred_patient_index.append(test[i])
                local_patient_index.append(i)
                
                
         
            
        classifier_2.fit(feature_pwave[train],y_arr[train])
        
        pwave_y_test=feature_pwave[rr_pred_patient_index]
        #print ("rr_y_test is:  " + str(rr_y_test))
        y_predicted_pwave=classifier_2.predict(pwave_y_test)
         
        
        for i in range(len(y_predicted_pwave)):
            if y_predicted_pwave[i] == 0:
                pwave_pred_normal_index.append(local_patient_index[i])
            elif y_predicted_pwave[i] ==1:
                pwave_pred_patient_index.append(local_patient_index[i])
                 
        
       
        
    #     for val in rr_pred_normal_index:
    #         if y_predicted_pwave[val] == y_predicted_rr[val]:
    #             y_predicted[val]=0;
    #         else:
    #             y_predicted[val]=1
        for val in local_index:
            if y_predicted[val] != -1:
                print("error in local _index loop")
                #exit()
            
            y_predicted[val]=0;
        
        for val in pwave_pred_normal_index:
            if y_predicted[val] != -1:
                print("error in pwave_pred_normal loop")
                #exit()
            y_predicted[val]=0;
       
        for val in pwave_pred_patient_index:
            if y_predicted[val] != -1:
                print('val is : ' + str(val))
                print('y predicted[val] is : ' + str(y_predicted[val]))
                print("error in pwave_pred_patient loop")
                #exit()
            #if y_predicted_pwave[val] != y_predicted_rr[val]:
            y_predicted[val]=1;
            
        
        y_predicted=y_predicted_rr
        
        
        ##make predicted array
        #print y_predicted
         
        y_predicted_report.extend(y_predicted)
        y_test_report.extend(y_arr[test])
         
         
     
     
    print y_test_report
    print y_predicted_report
    ####### Compute confusion matrix #######
    cm = metrics.confusion_matrix(y_test_report, y_predicted_report)
    np.set_printoptions(precision=2)
    print('Confusion matrix for: rf_sklearn')
    print(cm)
     
       
    print ("overall accuracy score of the classifier is")
    print metrics.accuracy_score(y_test_report, y_predicted_report)
    target_names = ['class 0', 'class 1']
    print(classification_report(np.array(y_test_report), np.array(y_predicted_report), target_names=target_names));
####################################################
exit()

###################################################
### simple cross validation here

# Run classifier with cross-validation and plot ROC curves

folds=10
cv = StratifiedKFold(y, n_folds=folds)
classifier = RandomForestClassifier(n_estimators=100)

y_test_report=[];
y_predicted_report=[]
 
 
all_indexes=[]
index_list=[]
for i, (train, test) in enumerate(cv):
 
     
    y_predicted=[]
    classifier.fit(feature_matrix[train], y_arr[train]) 
    y_predicted=(classifier.predict(feature_matrix[test]))
    y_test_report.extend(y_arr[test])
    y_predicted_report.extend(y_predicted)


print y_test_report
print y_predicted_report
####### Compute confusion matrix #######
cm = metrics.confusion_matrix(y_test_report, y_predicted_report)
np.set_printoptions(precision=2)
print('Confusion matrix for: rf_sklearn')
print(cm)

  
print ("overall accuracy score of the classifier is")
print metrics.accuracy_score(y_test_report, y_predicted_report)
target_names = ['class 0', 'class 1']
print(classification_report(np.array(y_test_report), np.array(y_predicted_report), target_names=target_names));

####################################################
exit()




