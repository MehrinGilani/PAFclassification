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
import wfdb_setup as ws;
import process_rr as pr;
import data_cleaning as dc;
import graphs 
import read_write as rw;
import non_linear_measures as nlm;
import classification_functions as cl
import sklearn

##############################################################################

output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/5min_trend_features/"
#output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/afpdb_test_records/"

### read values from text files ######
all_features=rw.read_features_frm_file(output_folder,"all_features_pickle.txt")
rw.write_value(all_features,output_folder,"list_of_list_before_cleaning","w")
global_vocab=rw.read_features_frm_file(output_folder,"global_vocab_pickle.txt")
rec_name_array=rw.read_features_frm_file(output_folder,"rec_name_array_pickle.txt")


#selected_feature_index=rw.read_features_frm_file(output_folder,"rfecv_selected13_features.txt")
##################### change key value pairs of global vocab ####################
inv_global_vocab = dict(zip(global_vocab.values(), global_vocab.keys()))
#print type(inv_global_vocab.values())
all_features_list=inv_global_vocab.values()
np.savetxt(output_folder+"all_features_list.txt",all_features_list,fmt="%s",delimiter=',',newline='\n')


#generate class labels
y=np.array(cl.generate_labels(rec_name_array))
print ("label array is: " + str(y))


#convert list of lists to matrix
all_feature_matrix=cl.covert_array_to_matrix(all_features,len(all_features));

#print all_feature_matrix
print ("shape of all feature matrix  is: " + str(all_feature_matrix.shape))
#################### SEPARATING EVALUATION DATA #########################
#X_cv, X_eval, y_cv, y_eval = cross_validation.train_test_split(all_feature_matrix, y, test_size=0.2, random_state=0)

exit()

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

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
folds=10
cv = StratifiedKFold(y, n_folds=folds,shuffle=True)
classifier = svm.SVC(kernel='linear', probability=True)
classifier = svm.SVC(kernel='linear', probability=True)

# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []


all_indexes=[]
index_list=[]

y_test_report=[];
y_predicted_report=[]
y_proba_report=[]

for i, (train, test) in enumerate(cv):
    ## prepare and normalize test train matrices
    
    print(train)
    exit()
    
    normalized_matrix_train=cl.normalise_mean_var(all_feature_matrix[train])
    normalised_matrix_test=cl.normalise_mean_var(all_feature_matrix[test])
    
    y_predicted2=[]
    
    #select features using rfecv only on train data
    only_feature_selection_matrix,index_arr_onlyfs=cl.select_optimal_features(normalized_matrix_train,y[train],classifier)
    
    #index_num,index_freq=cl.sort_and_combine_feature_indices(index_arr_onlyfs)
    for val in index_arr_onlyfs:
        #print ("val is: " +str(val))
        print (inv_global_vocab[val])
    
    
    #index_num_fs_only,index_freq_fs_only=cl.sort_and_combine_feature_indices(index_arr_onlyfs)
    
    
    
    matrix_for_train=cl.make_new_matrix(index_arr_onlyfs,normalized_matrix_train)  
    #classifier.fit(matrix_for_train, y[train]) 
    
    matrix_for_test=cl.make_new_matrix(index_arr_onlyfs,normalised_matrix_test)  
    probas_ = classifier.fit(matrix_for_train, y[train]).predict_proba(matrix_for_test)
    ##########  ADDING VARIABLES FOR CLASSIFICATION REPORT HERE ####################
   
    
    
    y_proba_report.extend(probas_)
    y_predicted2=(classifier.predict(matrix_for_test))     
    print("f1-score for this set of features is:  "+ str(f1_score(y[test],y_predicted2)))
    y_predicted_report.extend(y_predicted2)
    y_test_report.extend(y[test])  
    
    index_list.append(index_arr_onlyfs)

#########################################################
#### write to file ######

######## Accumulating indexes of features selected in each fold #############
# 
# index_num,index_freq=cl.sort_and_combine_feature_indices(index_list)
# #print("index numbers are: " + str(index_num))
# #print("index freq are: " + str(index_freq))
# print ("total number of features selected after rfecv in cross validation: " + str(len(index_num)))
# 
# ####### Saving accmulated feature names in text file #######################
# accumulated_feature_arr=[]
# for val in index_num:
#     #print ("val is: " +str(val))
#     #print (inv_global_vocab[val])
#     accumulated_feature_arr.append(inv_global_vocab[val])
# 
# 
# np.savetxt(output_folder+"accumulated_features.txt",accumulated_feature_arr,fmt="%s",delimiter=',',newline='\n')

#this function adds the feature value to the comma separated file
# f=open(output_folder+"exel.txt",'w'); 
# for val in accumulated_feature_arr:
#     f.write(str(val)+" ")
# 
# for val in y_proba_report:
#     f.write(str(val)+" ")
#     
# for val in y_predicted_report:
#     f.write(str(val)+" ")
# 
# for val in y_test_report:
#     f.write(str(val)+" ")



###### Compute confusion matrix and classsfication report  #######
print "confusion matrix of cv loop is" 
cl.print_confusion_matrix(y_test_report, y_predicted_report,"rfecv")
cl.print_classification_report(y_test_report, y_predicted_report,['class 0', 'class 1'])

 
#+str(y_proba_report)+" " + str(y_predicted_report) +" " + str(y_test_report))

#np.savetxt(output_folder+"accumulated_features.txt",accumulated_feature_arr,fmt="%s",delimiter=',',newline='\n')

#np.savetxt(output_folder+"all_features_list.txt",all_features_list,fmt="%s",delimiter=',',newline='\n')
print "###################################"

###################################################################3
# #################### fitting the classifier with features selected using rfecv#########################
# cv_eval = StratifiedKFold(y_eval, n_folds=folds)
# 
# y_test_report_accumulated=[];
# y_predicted_report_accumulated=[]
# y_predicted_report_accumulated,y_test_report_accumulated=cl.do_cross_validation(classifier,cv_eval,X_eval,y_eval,index_num)
# 
# 
# ####### Compute confusion matrix and classsfication report  #######
# #print("y_predicted after extending is : "+ str(y_predicted2))  
# #print("y_test reportafter extending is : "+ str(y_test_report))  
# cl.print_confusion_matrix(y_test_report_accumulated, y_predicted_report_accumulated,"accumulated")
# cl.print_classification_report(y_test_report_accumulated, y_predicted_report_accumulated,['class 0', 'class 1'])

exit()





