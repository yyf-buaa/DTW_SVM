import datetime
import logging

import numpy as np
import pandas as pd

from geopy.distance import geodesic
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#SVM4：采用全程的平均速度作为距离度量
a = datetime.datetime.now()
b = 'SVM4:'+a.strftime('%Y-%m-%d %H-%M-%S')
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO,filename='log/'+b+'.log',filemode='w')
#reference:Dynamic Time-Alignment Kernel in Support Vector Machine
def dtw_kernel(X,Y):
    alpha = 1
    x_sample_num = len(X)
    y_sample_num = len(Y)
    result = np.zeros(shape = [x_sample_num,y_sample_num])

    for i in tqdm(range(x_sample_num)):
        for t in range(y_sample_num):
            x = X[i][0][0]
            y = Y[t][0][0]
            result[i][t] = abs(x-y)
            result[i][t] = np.exp(-result[i][t]/alpha)

    return result

clf = svm.SVC(kernel=dtw_kernel,C=10)
max_traj_len = 100
total_traj_num = 5000
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
        flag = True
        X = np.zeros([1,1])
        #计算平均速度
        x1 = df.loc[0,"Latitude"]
        y1 = df.loc[0,"Longitude"]
        x2 = df.loc[l-1,"Latitude"]
        y2 = df.loc[l-1,"Longitude"]
        distance = geodesic((x1,y1),(x2,y2)).m
        t1_str = df.loc[0,"Time"]
        d1_str = df.loc[0,"Date"]
        t2_str = df.loc[l-1,"Time"]
        d2_str = df.loc[l-1,"Date"]
        t1 = datetime.datetime(*map(int,d1_str.split('-')),*map(int,t1_str.split(':')))
        t2 = datetime.datetime(*map(int,d2_str.split('-')),*map(int, t2_str.split(':')))
        duration = (t2-t1).seconds
        X[0][0] = distance/duration
        if flag:
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


