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
from sklearn.metrics import accuracy_score,precision_score,recall_score
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

##############################################################################
output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/non_linear_features_edges_changed/"

#output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/afpdb_test_records/"

### read values from text files ######
all_features=rw.read_features_frm_file(output_folder,"all_features_pickle.txt")
rw.write_value(all_features,output_folder,"list_of_list_before_cleaning","w")
global_vocab=rw.read_features_frm_file(output_folder,"global_vocab_pickle.txt")
rec_name_array=rw.read_features_frm_file(output_folder,"rec_name_array_pickle.txt")

##################### change key value pairs of global vocab ####################
inv_global_vocab = dict(zip(global_vocab.values(), global_vocab.keys()))
#print type(inv_global_vocab.values())
all_features_list=inv_global_vocab.values()
np.savetxt(output_folder+"all_features_list.txt",all_features_list,fmt="%s",delimiter=',',newline='\n')


#generate class labels
y=np.array(cl.generate_labels(rec_name_array))
print ("label array is: " + str(y))


#convert list of lists to matrix
all_feature_matrix=cl.covert_array_to_matrix(all_features,len(all_features),max(global_vocab.values())+1);

#print all_feature_matrix
#print ("type of all feature matrix  is: " + str(type(all_feature_matrix)))


#################### SEPARATING EVALUATION DATA #########################
X_cv, X_eval, y_cv, y_eval = cross_validation.train_test_split(all_feature_matrix, y, test_size=0.2, random_state=0)

X_cv_normalized_matrix=cl.normalise_mean_var(X_cv)

X_eval_normalized_matrix=cl.normalise_mean_var(X_eval)

############## with normalisation ######################
# Classification
normalised="  "
normalized_matrix=cl.normalise_mean_var(all_feature_matrix)
#print ("type of normalised_matrix is: " + str(type(normalized_matrix)))
#print("normalized matrix is: ")
#print normalized_matrix
#rw.write_value(normalized_matrix,output_folder,"all_features_normalized.txt",'w')

#feature_matrix=normalized_matrix;
#feature_matrix=all_feature_matrix

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves


# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []


all_indexes=[]
index_list=[]


###############################  FEATURE SELECTION ################################################

######## Feature selection using cross-validation and recursive feature elimination #############

# for i, (train, test) in enumerate(cv):
#      
#     normalized_matrix_train=cl.normalise_mean_var(all_feature_matrix[train])
#     normalised_matrix_test=cl.normalise_mean_var(all_feature_matrix[test])
#      
#     y_predicted=[]
#     selected_features,index_arr=cl.select_optimal_features(normalized_matrix_train,y[train],classifier)
#     print("shape of selected_features_for training is: " + str(selected_features.shape))
#     print("shape of y_arr_for training is: " + str(y[train].shape))
#  
#     index_list.append(index_arr)



######## Accumulating indexes of features selected in each fold #############

index_num,index_freq=cl.sort_and_combine_feature_indices(index_list)
print("index numbers are: " + str(index_num))
print("index freq are: " + str(index_freq))
print ("total number of features selected after rfecv in cross validation: " + str(len(index_num)))

####### Saving accmulated feature names in text file #######################
accumulated_feature_arr=[]
for val in index_num:
    #print ("val is: " +str(val))
    #print (inv_global_vocab[val])
    accumulated_feature_arr.append(inv_global_vocab[val])

np.savetxt(output_folder+"accumulated_features.txt",accumulated_feature_arr,fmt="%s",delimiter=',',newline='\n')
####################################################################################



#################### fitting the classifier with features selected using rfecv in cross_validation#########################
# y_test_report_rfecv_cv=[];
# y_predicted_report_rfecv_cv=[]
# y_predicted_report_rfecv_cv,y_test_report_rfecv_cv=cl.do_cross_validation(classifier,cv,all_feature_matrix,y,index_num)
#  
# ####### Compute confusion matrix and classsfication report  #######
# cl.print_confusion_matrix(y_test_report_rfecv_cv, y_predicted_report_rfecv_cv,"rfecv_cvloop")
# cl.print_classification_report(y_test_report_rfecv_cv, y_predicted_report_rfecv_cv,['class 0', 'class 1'])

print("############################################")

folds=5
cv = StratifiedKFold(y_eval, n_folds=folds)
classifier = svm.SVC(kernel='linear', probability=True)

###################### Feature selection using RFECV only #################################################

only_feature_selection,index_arr_onlyfs=cl.select_optimal_features(X_cv_normalized_matrix,y_cv,classifier)
print("number of features selected only with rfecv: " +str(len(index_arr_onlyfs)))
index_num_fs_only,index_freq_fs_only=cl.sort_and_combine_feature_indices(index_arr_onlyfs)
print("index numbers are: " + str(index_num_fs_only))

rw.write_features_to_file(index_num_fs_only,output_folder,"rfecv_selected13_features.txt")
#print("index freq are: " + str(index_freq_fs_only))


####### print features selected by rfecv alone #########################################
rfecv_only_feature_arr=[]
for val in index_num_fs_only:
    #print val
    #print (inv_global_vocab[val])
    rfecv_only_feature_arr.append(inv_global_vocab[val])
    
print("len of rfecv only features is: " +str(len(rfecv_only_feature_arr)))
np.savetxt(output_folder+"rfecv_only_features.txt",rfecv_only_feature_arr,fmt="%s",delimiter=',',newline='\n')

#######################################################################
#plt.figure()
#plt.title("frequency of indexes")
#plt.scatter(index_num,index_freq)
#plt.figure()

#################### fitting the classifier with features selected using rfecv#########################
y_test_report_rfecv=[];
y_predicted_report_rfecv=[]
y_predicted_report_rfecv,y_test_report_rfecv=cl.do_cross_validation(classifier,cv,X_eval,y_eval,index_num_fs_only)

    
#########################################################
####### Compute confusion matrix and classsfication report  #######
cl.print_confusion_matrix(y_test_report_rfecv, y_predicted_report_rfecv,"rfecv")
cl.print_classification_report(y_test_report_rfecv, y_predicted_report_rfecv,['class 0', 'class 1'])
#########################################################

plt.show()
exit()




