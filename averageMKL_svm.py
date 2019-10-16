#coding=utf-8
'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-16 14:40:06
@LastEditTime: 2019-10-16 19:20:14
@LastEditors: Please set LastEditors
'''
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from combine_kernels import combine_kernels
from sklearn.metrics.pairwise import rbf_kernel
import argparse

def averge_svm(train_x,train_y,test_x,test_y,c):
   
    K_train = np.array(train_x) #这时K_train是三维矩阵了
    K_test = np.array(test_x)  #这时K_test是三维矩阵了
    m=K_train.shape[0]
    kernel_weights = np.ones((m, 1))
    kernel_weights = kernel_weights / m
    kernel_weights=kernel_weights.tolist()
    print('weights of kernels:',kernel_weights)

    K_train_com = combine_kernels(kernel_weights, K_train)
    K_test_com = combine_kernels(kernel_weights, K_test)

    K_train_com.tolist()
    K_test_com.tolist()
    # 3.train and test model
    #print '开始训练'
    train_y = train_y.reshape(len(K_train_com),).tolist()
    clf = svm.SVC(C=c,kernel='precomputed',probability=True)
    clf.fit(K_train_com, train_y)
    #print '开始预测'
    y_pred = clf.predict(K_test_com)
    y_pred_proba=clf.predict_proba(K_test_com)
    y_pred_proba=y_pred_proba[:,1]
    return y_pred.tolist(),y_pred_proba.tolist()
def getopt():
    parse=argparse.ArgumentParser()
    parse.add_argument('-l','--label',type=str,default=None)
    parse.add_argument('-cv','--crossvalid',type=int,default=5)
    args=parse.parse_args()
    return args
if __name__ == "__main__":
    
    #getparam
    args=getopt()
    label_path=args.label
    cv=args.crossvalid
    feature_path='./feature'
    conf=np.loadtxt('gamma.csv',delimiter=',',dtype=str)
    feature_name_list=conf[0].tolist()
    gamma_list=conf[1].tolist()
    gamma_list=[float(e) for e in gamma_list]
    

    #load_feature
    n_feature=len(feature_name_list)
    print n_feature
    
    feature_list=[]
    for feature_name in feature_name_list:
        feature=np.loadtxt('feature/'+feature_name,delimiter=',',dtype=float)
        feature_list.append(feature)
    n_sample=feature_list[0].shape[0]
    
    indices=np.arange(n_sample)
    np.random.shuffle(indices)
    #load labels
    if label_path:
        labels=np.loadtxt(label_path,dtype=int)
    else:
        labels=np.array([1]*int(n_sample/2)+[-1]*int(n_sample/2))
    #labels=labels[indices]
    #compute kernel
    K_feature_list=[]
    for i in range(n_feature):
        print feature_list[i].shape
        K_feature_list.append(rbf_kernel(feature_list[i],gamma=gamma_list[i]))

    #cross validation 
    y_pred_list=list()
    y_true_list=list()
    y_proba_list=list()
    skflod=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    
    c_map=map(lambda x:2**x,np.linspace(-2,5,7))
    c_list=list(c_map)
    result=[]
    for c in c_list:
        for train,test in skflod.split(K_feature_list[0],labels):
            train_data=[k[train] for k in K_feature_list]
            train_data=[k[:,train] for k in train_data]
            test_data=[k[test] for k in K_feature_list]
            test_data=[k[:,train]for k in test_data]

            train_labels=labels[train]
            test_labels=labels[test]
            y_pred,y_proba=averge_svm(train_data,train_labels,test_data,test_labels,c)
            y_pred_list.extend(y_pred)
            y_proba_list.extend(y_proba)
            y_true_list.extend(test_labels)
        acc=metrics.accuracy_score(y_true_list,y_pred_list)
        roc_auc=metrics.roc_auc_score(y_true_list,y_proba_list)
        print 'c:',c,'acc:',acc,'roc_auc:',roc_auc
        print '\n'
        result.append([c,acc,roc_auc])
    np.savetxt('averageMKL_result.csv',result,delimiter=',',fmt='%5f')
    








