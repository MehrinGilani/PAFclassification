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
#folder to save csv in 
output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afpdb_apdb_nsrdb/"

#folder to read feartures from
output_folder_afpdb="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afpdb_apdb_nsrdb/afpdb/"
output_folder_nsrdb="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afpdb_apdb_nsrdb/nsrdb/"
output_folder_afdb="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afpdb_apdb_nsrdb/afdb/"

### read values from text files ######

## read these once
global_vocab=rw.read_features_frm_file(output_folder_afpdb,"global_vocab_pickle.txt")


## read features from different databases

all_features_afpdb=rw.read_features_frm_file(output_folder_afpdb,"all_features_pickle.txt")
all_features_nsrdb=rw.read_features_frm_file(output_folder_nsrdb,"all_features_pickle.txt")
all_features_afdb=rw.read_features_frm_file(output_folder_afdb,"all_features_pickle.txt")

## read rec_name_array
rec_name_array_afpdb=rw.read_features_frm_file(output_folder_afpdb,"rec_name_array_pickle.txt")
rec_name_array_nsrdb=rw.read_features_frm_file(output_folder_nsrdb,"rec_name_array_pickle.txt")
rec_name_array_afdb=rw.read_features_frm_file(output_folder_afdb,"rec_name_array_pickle.txt")

all_rec_name_array=rec_name_array_afpdb+rec_name_array_nsrdb+ rec_name_array_afdb
print ("all rec_name array is: " + str(all_rec_name_array))
#generate class labels
y_afpdb=cl.generate_labels("afpdb",rec_name_array_afpdb)
y_nsrdb=cl.generate_labels("nsrdb",rec_name_array_nsrdb)
y_afdb=cl.generate_labels("afdb",rec_name_array_afdb)

y_all=np.array(y_afpdb+y_nsrdb+y_afdb)
print ("all label array is: " + str(y_all))

##################### change key value pairs of global vocab ####################
inv_global_vocab = dict(zip(global_vocab.values(), global_vocab.keys()))
#print type(inv_global_vocab.values())
all_features_list=inv_global_vocab.values()
np.savetxt(output_folder+"all_features_list.txt",all_features_list,fmt="%s",delimiter=',',newline='\n')


#combine lists and convert list of lists to one big matrix
all_feature_matrix=cl.covert_array_to_matrix(all_features_afpdb+all_features_nsrdb+all_features_afdb);

normalized_matrix=cl.normalise_mean_var(all_feature_matrix)
#print all_feature_matrix
print ("shape of all feature matrix  is: " + str(all_feature_matrix.shape))

#################### GENERATE TRAIN TEST INDICES FOR SHUFFLE SPLIT #########################
#X_cv, X_eval, y_cv, y_eval = cross_validation.train_test_split(all_feature_matrix, y, test_size=0.2, random_state=0)

cv_shufflesplit=cross_validation.ShuffleSplit(len(y_all),1,test_size=0.2,train_size=None, random_state=0)

#################### Save feaures and y_all to csv file #########################
csv_indexes=sorted(inv_global_vocab.keys())
#print(csv_indexes)
csv_header=[]
for val in csv_indexes:
    f_name=inv_global_vocab[val]
    csv_header.append(f_name)

# rw.combine_n_write_df_to_csv(all_feature_matrix, csv_header, y,["labels"],output_folder, "features_NOT_normalised_afib_patient.csv")
# 
# exit()


#rw.write_df_to_csv(all_feature_matrix, csv_header, output_folder, "features_NOT_normalised_test.csv")
#rw.write_df_to_csv(y,["labels"],output_folder,"label.csv")
# rw.combine_n_write_df_to_csv(all_feature_matrix, csv_header, y,["labels"],output_folder, "features_NOT_normalised_learning_Xtrain_ss_testtttt.csv")
# exit()
for train_index, test_index in cv_shufflesplit:
    print("%s %s" % (train_index, test_index))
    test_recs=[]
    train_recs=[]
    ################### PICK REC NAMES CORRESPONDING TO TRAIN TEST INDEX ############
    for ind_test in test_index:
        test_recs.append(all_rec_name_array[ind_test])
    for ind_train in train_index:
        train_recs.append(all_rec_name_array[ind_train])
    #combine_n_write_df_to_csv(feature_matrix,feature_names,col_to_add,added_col_header,output_folder,file_name)
    rw.combine_n_write_df_to_csv(all_feature_matrix[train_index], csv_header, y_all[train_index],["labels"],output_folder, "RR_features_3dbs_Xtrain.csv")
    rw.combine_n_write_df_to_csv(train_index,["training_index"],train_recs,["training_recs"],output_folder,"RR_train_index_n_recname.csv")
    rw.combine_n_write_df_to_csv(all_feature_matrix[test_index], csv_header,y_all[test_index],["labels"], output_folder, "RR_features_3dbs_Xtest.csv")
    rw.combine_n_write_df_to_csv(test_index,["testing_index"],test_recs,["testing_recs"],output_folder,"RR_test_index_n_recname.csv")

exit()























































