import datetime
import logging

import numpy as np
import pandas as pd
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from matplotlib.pylab import plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from geopy.distance import geodesic
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#SVM1：采用经纬度做欧式距离的DTW-Kernel
a = datetime.datetime.now()
b = 'SVM1:'+a.strftime('%Y-%m-%d %H-%M-%S')
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO,filename='log/'+b+'.log',filemode='w')
#reference:Dynamic Time-Alignment Kernel in Support Vector Machine
def dtw_kernel(X,Y):
    """
    :param X:[ [[x1,y1],[x2,y2],[x3,y3]........[xn,yn]].......]
    :param Y:[ [[x1,y1],[x2,y2],[x3,y3]........[xm,ym]]......]
    :return: dtw_similarity_matrix
    """
    alpha = 100000
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
            G[0,1:] = float("inf")
            G[1:,0] = float("inf")
            for j in range(1,x_len+1):
                for k in range(1,y_len+1):
                    #这里是将序列数据向后对齐一格
                    value1 = G[j-1][k]+geodesic((x[j-1][0],x[j-1][1]),(y[k-1][0],y[k-1][1])).m
                    value2 = G[j-1][k-1]+2*geodesic((x[j-1][0],x[j-1][1]),(y[k-1][0],y[k-1][1])).m
                    value3 = G[j][k-1]+geodesic((x[j-1][0],x[j-1][1]),(y[k-1][0],y[k-1][1])).m
                    G[j][k] = min([value1,value2,value3])
            result[i][t] = G[x_len][y_len]/(x_len+y_len)
            result[i][t] = np.exp(-result[i][t]/alpha)

    return result

clf = svm.SVC(kernel=dtw_kernel,C=10)
max_traj_len = 30
total_traj_num = 10
logging.info("max trajactory length:{}".format(max_traj_len))
logging.info("total trajactory number:{}".format(total_traj_num))
X_total = []
Y_total = []
total_num = 0
positive_num = 0
for i in range(total_traj_num):
    df = pd.read_csv('traj/{}.csv'.format(i))
    if len(df)>0 and len(df)<max_traj_len:
        l = len(df)
        X = np.zeros([l,2])
        for j in range(l):
            X[j][0] = df.loc[j,"Latitude"]
            X[j][1] = df.loc[j,"Longitude"]
        if df.loc[1,"mode"] == "taxi" or df.loc[1,"mode"] == "car":
            Y_total.append(1)
            positive_num=positive_num+1
            total_num = total_num + 1
            X_total.append(X)
        if df.loc[1, "mode"] == "walk":
            Y_total.append(0)
            total_num = total_num + 1
            X_total.append(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size = 0.2, random_state = 0)
print(total_num)
print(positive_num)
logging.info("finish split train and test dataset")
logging.info("trainset length:{}".format(len(X_train)))
logging.info("testset length:{}".format(len(X_test)))
clf.fit(X_train,Y_train)
logging.info("finish fit")
Y_pred_train = clf.predict(X_train)
score = accuracy_score(Y_train,Y_pred_train)
logging.info("accuracy on trainset:{}".format(score))
Y_pred = clf.predict(X_test)
logging.info("finish predict")
score = accuracy_score(Y_test,Y_pred)
logging.info("accuracy on testset:{}".format(score))
cm = confusion_matrix(Y_test,Y_pred)
print('Confusion matrix\n\n',cm)
logging.info("")
logging.info(classification_report(Y_test,Y_pred))


