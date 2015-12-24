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
from numpy import dtype, argmin, NAN
from scipy.stats.mstats_basic import kurtosistest
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing;

from sklearn.feature_selection import RFECV
import pickle 
import collections
from sklearn.cross_validation import train_test_split,StratifiedKFold,ShuffleSplit
from sklearn.metrics.metrics import confusion_matrix ,accuracy_score,classification_report,f1_score
import wfdb_setup as ws;
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.svm import SVC
import read_write as rw;


def fill_global_vocab(feature_name,index_of_features,feature_dictionary):
    if feature_name not in feature_dictionary.keys():
        feature_dictionary[feature_name]=index_of_features;
        index_of_features=index_of_features+1;
    return(feature_dictionary,index_of_features)
        


def generate_labels(db_name,rec_array):
    label_array=[]
    if db_name == "afpdb":
        for val in rec_array:
            if "n" in val:
                label_array.append("normal")
            elif "p" in val:
                label_array.append("patient")
            elif "t" in val:
                label_array.append("?")
    
    if db_name =="nsrdb":
        for val in rec_array:
            label_array.append("normal")
    
    if db_name =="afdb":
        for val in rec_array:
            label_array.append("patient")
    
    print("done writing labels")
    
    return label_array

        
        
def covert_array_to_matrix(array):
    converted_matrix=np.matrix(array)
    return converted_matrix



def normalise_mean_var(feature_matrix):
    #normalise mean and variance
    #check the effect of normalised mean
    normalized_matrix=preprocessing.scale(feature_matrix,axis=0)
    return normalized_matrix

def get_features_from_folder(output_folder):
    ### read values from text files ######
    all_features=rw.read_features_frm_file(output_folder,"all_features_pickle.txt")
    rec_name_array=rw.read_features_frm_file(output_folder,"rec_name_array_pickle.txt")
    return all_features,rec_name_array

def train_classifier(feature_list,label_array,cross_validation_type):
# train a simple svm main or linear cflassifier
#correct return values
    result="result";
    return result


def select_optimal_features(feature_matrix,y,classifier):
    
    #print("type of cv is: " + str(cv))
    ################################## preparing feature matirx with optimal features ############################
    #reduced_data = PCA(n_components=25).fit_transform(feature_matrix)
    
    #print("shape of reduced data before rfecv is: " +str(reduced_data.shape))
    # Create the RFE object and compute a cross-validated score.
    #classifier = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    cv=StratifiedKFold(y, 5)
    #print("type of cv is: " + str(cv))
    
    rfecv = RFECV(estimator=classifier, step=1, cv=cv,scoring='accuracy')
    print("going to select optimal features")
    rfecv.fit(feature_matrix, y)
    print("done selecting optimal features")
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    ## ranking_ : array of shape [n_features]
    #The feature ranking, such that ranking_[i] corresponds to the ranking position of the i-th feature. 
    #Selected (i.e., estimated best) features are assigned rank 1.
    #print("shape of reduced data after rfecv is: " +str(reduced_data.shape))
    #print("ranking list is: " + str(rfecv.ranking_))
    #print(type(rfecv.ranking_))
    ranked_features=rfecv.ranking_.tolist()
    
    index=[]
    for i in range(0,len(ranked_features)):
        if ranked_features[i] is 1:
            index.append(i)
        
    print("index is"+str(index))
    
       
    i=0;
    selected_features=np.zeros(shape=(len(feature_matrix), len(index)),dtype=np.float64); #initialze with zeros
    for val in index:
        selected_features[:,i]=feature_matrix[:,val]
        i=i+1
    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    
#     print((selected_features.shape))
#     path_to_file="/home/ubuntu/Documents/Data_challenge/dc_3/dc_3_try2/"
#     file_name="selected_features"  
#     with open(path_to_file+file_name,"w") as internal_filename:
#             pickle.dump(selected_features,internal_filename)

    return selected_features,index

def make_new_matrix(index,old_matrix):
    i=0;
    selected_features=np.zeros(shape=(len(old_matrix), len(index)),dtype=np.float64); #initialze with zeros
    for val in index:
        selected_features[:,i]=old_matrix[:,val]
        i=i+1
    return selected_features


def sort_and_combine_feature_indices(index_listoflist):
    #this feature takes in list of list and sorts, and combines and gives one list of features ranked as 1 in each fold 
    ######## plotting frequency of index lists #############
    all_indexes=[]
    for array_list in index_listoflist:
        #print type(array_list)
        if isinstance(array_list, int):
            all_indexes.append(array_list)
        else:
            for val in array_list:
                all_indexes.append(val)
    
    
    all_indexes.sort()
    #print ("all_indexes is :  " + str(all_indexes))
    
    unique_index = list(set(all_indexes))
    
    index_num=[]
    index_freq=[]
   
    for x in unique_index:
        index_num.append(x)
        index_freq.append(all_indexes.count(x))
    print ("index_num returned after sort n combine is :  " + str(index_num))
    return index_num,index_freq
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learn_curve(classifier,cv,feature_matrix,y):
    ###################LEARNING CURVES ######################
    title = "Learning Curves (SVC, $\gamma=0.001$)"
#     print y
#     cls, y = np.unique(y, return_inverse=True)
#     print len(cls)
    # SVC is more expensive so we do a lower number of CV iterations:
    #cv = cross_validation.ShuffleSplit(len(y), n_iter=10,test_size=0.2, random_state=0)
    #cv=StratifiedKFold(y, n_folds=5)
#     estimator = svm.SVC(kernel='linear',gamma=0.001)
    estimator=classifier
    plot_learning_curve(estimator, title, feature_matrix, y, (0.7, 1.01), cv=cv)
    return None

def do_cross_validation(classifier,cv,all_feature_matrix,y,indexes_of_selected_features):    
    y_test_report=[];
    y_predicted_report=[]
    #score_array=[]
    for i, (train, test) in enumerate(cv):
        normalized_matrix_train=normalise_mean_var(all_feature_matrix[train])
        normalised_matrix_test=normalise_mean_var(all_feature_matrix[test])
        
        y_predicted2=[]
        matrix_for_train=make_new_matrix(indexes_of_selected_features,normalized_matrix_train)  
        classifier.fit(matrix_for_train, y[train]) #if this gives error then remove train from selected_features[train]
        
        matrix_for_test=make_new_matrix(indexes_of_selected_features,normalised_matrix_test)  
        probas_ = classifier.fit(matrix_for_train, y[train]).predict_proba(matrix_for_test)
        #print("shape of selected_features_for second cross validation loop  is: " + str(matrix_for_test.shape))
        ##########  ADDING VARIABLES FOR CLASSIFICATION REPORT HERE ####################
        y_predicted2=(classifier.predict(matrix_for_test))     
        y_predicted_report.extend(y_predicted2)
        y_test_report.extend(y[test])    
        #f1_cv_score=f1_score(y[test], y_predicted2)
        #score_array.append(f1_cv_score)
    
    return y_predicted_report,y_test_report    

def print_confusion_matrix(y_test_report, y_predicted_report,algo_name):
    #y_predicted report is one array of len(100) , this array has all the predicted values of all folds
    cm = confusion_matrix(y_test_report, y_predicted_report)
    np.set_printoptions(precision=2)
    print('Confusion matrix for: ' +algo_name)
    print(cm)
    return None

def print_classification_report(y_test_report, y_predicted_report,target_names):
    #target_names = ['class 0', 'class 1']
    print ("overall accuracy score of the classifier is")
    print accuracy_score(y_test_report, y_predicted_report)
    print(classification_report(np.array(y_test_report), np.array(y_predicted_report), target_names=target_names));
    return None


def plot_learning_curve_test(train_score,validation_score):
    plt.figure();
    plt.plot(range(len(train_score)),train_score,color='r')
    plt.plot(range(len(validation_score)),train_score,color='b')
    train_score
    return None
         