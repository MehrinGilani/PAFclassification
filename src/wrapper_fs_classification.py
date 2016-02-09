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
from sklearn.neighbors import KNeighborsClassifier as knn
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

##############################################################################
class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
        
##############################################################################
# #folder to read feartures from
output_folder="/home/ubuntu/Documents/Thesis_work/results/1_min_features/aftdb/"

output_folder_aftdb="/home/ubuntu/Documents/Thesis_work/results/1_min_features/aftdb/"
output_folder_afpdb_patient="/home/ubuntu/Documents/Thesis_work/results/1_min_features/afpdb_patient/"
output_folder_afpdb_normal="/home/ubuntu/Documents/Thesis_work/results/1_min_features/afpdb_normal/"


### read values from text files ######

## read these once
#global_vocab=rw.read_features_frm_file(output_folder_afpdb_normal,"global_vocab_pickle.txt")
global_vocab=rw.read_features_frm_file(output_folder,"global_vocab_pickle.txt")

# ##################### change key value pairs of global vocab ####################
inv_global_vocab = dict(zip(global_vocab.values(), global_vocab.keys()))
print inv_global_vocab[3] # confirm that this is the std_dev_features and del is from dictionary

del inv_global_vocab[3]
## read features from different databases
# 
all_features_aftdb=rw.read_features_frm_file(output_folder_aftdb,"all_features_pickle.txt")
all_features_afpdb_patient=rw.read_features_frm_file(output_folder_afpdb_patient,"all_features_pickle.txt")
all_features_afpdb_normal=rw.read_features_frm_file(output_folder_afpdb_normal,"all_features_pickle.txt")
# all_features_nsrdb=rw.read_features_frm_file(output_folder_nsrdb,"all_features_pickle.txt")
# all_features_afdb=rw.read_features_frm_file(output_folder_afdb,"all_features_pickle.txt")

#all_features=rw.read_features_frm_file(output_folder,"all_features_pickle.txt")
all_features=all_features_aftdb+all_features_afpdb_patient+all_features_afpdb_normal

# ## read rec_name_array
rec_name_array_aftdb=rw.read_features_frm_file(output_folder_aftdb,"rec_name_array_pickle.txt")
rec_name_array_afpdb_patient=rw.read_features_frm_file(output_folder_afpdb_patient,"rec_name_array_pickle.txt")
rec_name_array_afpdb_normal=rw.read_features_frm_file(output_folder_afpdb_normal,"rec_name_array_pickle.txt")
# rec_name_array_nsrdb=rw.read_features_frm_file(output_folder_nsrdb,"rec_name_array_pickle.txt")
# rec_name_array_afdb=rw.read_features_frm_file(output_folder_afdb,"rec_name_array_pickle.txt")
#rec_name_array=rw.read_features_frm_file(output_folder,"rec_name_array_pickle.txt")
all_rec_name_array=rec_name_array_aftdb+rec_name_array_afpdb_patient+ rec_name_array_afpdb_normal
#all_rec_name_array=rec_name_array
print ("all rec_name array is: " + str(all_rec_name_array))
print("size of all rec name array i.e num of egs is : " + str(len(all_rec_name_array)))
#generate class labels
# y_afpdb=cl.generate_labels("afpdb",rec_name_array_afpdb)
# y_nsrdb=cl.generate_labels("nsrdb",rec_name_array_nsrdb)
# y_afdb=cl.generate_labels("afdb",rec_name_array_afdb)
y_aftdb=cl.generate_labels_bool("aftdb",rec_name_array_aftdb)
y_afpdb_patient=cl.generate_labels_bool("afdb",rec_name_array_afpdb_patient)
y_afpdb_normal=cl.generate_labels_bool("nsrdb",rec_name_array_afpdb_normal)
y_all=np.array(y_aftdb+y_afpdb_patient+y_afpdb_normal)
#y_all=cl.generate_labels("afpdb", rec_name_array)
#print ("all label array is: " + str(y_all))


#convert list of lists to matrix
all_feature_matrix_old=cl.covert_array_to_matrix(all_features);
print ("shape of all feature matrix  is: " + str(all_feature_matrix_old.shape))
all_feature_matrix=np.delete(all_feature_matrix_old, 3, 1)

print ("shape of all feature matrix  is: " + str(all_feature_matrix.shape))

#################### SEPARATING EVALUATION DATA #########################
#X_cv, X_eval, y_cv, y_eval = cross_validation.train_test_split(all_feature_matrix, y, test_size=0.2, random_state=0)


###############################################################################
# Classification 

# Run classifier with cross-validation and plot ROC curves
#
folds=10
cv = StratifiedKFold(y_all, n_folds=folds,shuffle=True)
#cv_shufflesplit=cross_validation.ShuffleSplit(len(y_all),1,test_size=0.2,train_size=None, random_state=0)
#classifier = svm.SVC(kernel='linear', probability=True)
#classifier = RandomForestClassifierWithCoef(RandomForestClassifier)
classifier=knn(n_neighbors=3)

all_indexes=[]
index_list=[]

y_test_report=[];
y_predicted_report=[]
y_proba_report=[]

for i, (train, test) in enumerate(cv):
    ## prepare and normalize test train matrices    
    normalized_matrix_train=cl.normalise_mean_var(all_feature_matrix[train])
    normalised_matrix_test=cl.normalise_mean_var(all_feature_matrix[test])
    
    y_predicted2=[]
    
    #select features using rfecv only on train data
    #rfe = RFE(estimator=classifier, cv=5,n_features_to_select=10,step=2)
    rfe = RFECV(estimator=classifier, cv=5,step=2, scoring='f1')
    print("going to select optimal features")
    rfe.fit(normalized_matrix_train, y_all[train])
    ranked_features=(rfe.ranking_).tolist()
    #print("shape of train matrix after rfe.fit is: " +str(normalized_matrix_train.shape))
    index=[]
    for i in range(0,len(ranked_features)):
        if ranked_features[i] is 1:
            index.append(i)
        
    print("index is"+str(index))
    
    rfe.transform(normalized_matrix_train)
    #print("shape of transformed train matrix is: " +str(normalized_matrix_train.shape))
    classifier.fit(normalized_matrix_train,y_all[train])
    rfe.transform(normalised_matrix_test)
    #print("shape of transformed test matrix is: " +str(normalised_matrix_test.shape))
    probas_ = classifier.predict_proba(normalised_matrix_test)
    ##########  ADDING VARIABLES FOR CLASSIFICATION REPORT HERE ####################
   
    
    
    y_proba_report.extend(probas_)
    y_predicted2=(classifier.predict(normalised_matrix_test))     
    print("f1-score for this set of features is:  "+ str(f1_score(y_all[test],y_predicted2)))
    clf_score=classifier.score(normalised_matrix_test, y_all[test])
    print("score for this set of features is:  "+ str(clf_score))
    y_predicted_report.extend(y_predicted2)
    y_test_report.extend(y_all[test])  
    

    
    
    #index_num,index_freq=cl.sort_and_combine_feature_indices(index_arr_onlyfs)
#     for val in index_arr_onlyfs:
#         #print ("val is: " +str(val))
#         print (inv_global_vocab[val])
    
    
   
    
    #index_list.append(index_arr_onlyfs)

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

############# RFE on all training data ####################

# rfe_all = RFE(estimator=classifier, n_features_to_select=10,step=2)
# print("going to select optimal features from all training data")
# rfe_all.fit(all_feature_matrix, y_all)
# ranked_features=(rfe_all.ranking_).tolist()
# #print("shape of train matrix after rfe.fit is: " +str(normalized_matrix_train.shape))
# index=[]
# for i in range(0,len(ranked_features)):
#     if ranked_features[i] is 1:
#         index.append(i)
#     
# print("index is"+str(index))
#     
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





