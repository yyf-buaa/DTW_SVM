import datetime
import logging

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
a = datetime.datetime.now()
b = a.strftime('%Y-%m-%d %H-%M-%S')
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO,filename='log/'+b+'.log',filemode='w')
#reference:Dynamic Time-Alignment Kernel in Support Vector Machine
def dtw_kernel(X,Y):
    """
    :param X:[ [[x1,y1],[x2,y2],[x3,y3]........[xn,yn]].......]
    :param Y:[ [[x1,y1],[x2,y2],[x3,y3]........[xm,ym]]......]
    :return: dtw_similarity_matrix
    """
    x_sample_num = len(X)
    y_sample_num = len(Y)
    result = np.zeros(shape = [x_sample_num,y_sample_num])
    for i in tqdm(range(x_sample_num)):
        for t in range(y_sample_num):
            x = X[i]
            y = Y[t]
            x_len = x.shape[0]
            y_len = y.shape[0]
            G = np.zeros(shape=[x_len+1,y_len+1])
            for j in range(1,x_len+1):
                for k in range(1,y_len+1):
                    #这里是将序列数据向后对齐一格
                    value1 = G[j-1][k]+np.dot(x[j-1],y[k-1])
                    value2 = G[j-1][k-1]+2*np.dot(x[j-1],y[k-1])
                    value3 = G[j][k-1]+np.dot(x[j-1],y[k-1])
                    G[j][k] = max([value1,value2,value3])
            result[i][t] = G[x_len][y_len]/(x_len+y_len)
    return result

clf = svm.SVC(kernel=dtw_kernel,C=10,gamma=2.0)
max_traj_len = 20
total_traj_num = 10
logging.info("max trajactory length:{}".format(max_traj_len))
logging.info("total trajactory number:{}".format(total_traj_num))
X_total = []
Y_total = []
for i in range(total_traj_num):
    df = pd.read_csv('traj/{}.csv'.format(i))
    if len(df)>0:
        l = min(len(df),max_traj_len)
        X = np.zeros([l,2])
        for j in range(l):
            X[j][0] = df.loc[j,"Latitude"]
            X[j][1] = df.loc[j,"Longitude"]
        X_total.append(X)
        if df.loc[0,"mode"] == "taxi":
            Y_total.append(1)
        else:
            Y_total.append(0)
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size = 0.2, random_state = 0)
logging.info("finish split train and test dataset")
logging.info("trainset length:{}".format(len(X_train)))
logging.info("testset length:{}".format(len(X_test)))
svc = svm.SVC()
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['dtw_kernel']}]
grid_search = GridSearchCV(estimator=svc,param_grid=parameters,scoring='accuracy',cv=5,verbose=0)
grid_search.fit(X_train, Y_train)
logging.info('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
logging.info('Parameters that give the best results :','\n\n', (grid_search.best_params_))
logging.info('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, Y_test)))


