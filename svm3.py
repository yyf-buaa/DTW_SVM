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
#SVM3：短的采用全程平均速度补齐，长的直接截断，采用每一个采样间隔内的平均速度近似瞬时速度做DTW-Kernel
a = datetime.datetime.now()
b = 'SVM3:'+a.strftime('%Y-%m-%d %H-%M-%S')
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO,filename='log/'+b+'.log',filemode='w')
#reference:Dynamic Time-Alignment Kernel in Support Vector Machine
clf = svm.SVC(kernel="rbf",C=10,gamma="auto")
max_traj_len = 30
total_traj_num = 5000
logging.info("max trajactory length:{}".format(max_traj_len))
logging.info("total trajactory number:{}".format(total_traj_num))
X_total = []
Y_total = []
total_num = 0
positive_num = 0
for i in range(total_traj_num):
    X = np.zeros([max_traj_len - 1])
    df = pd.read_csv('traj/{}.csv'.format(i))
    flag = True
    if len(df)>0:
        if len(df)>=max_traj_len:
            #截断多余的
            for j in range(max_traj_len-1):
                # 计算平均速度
                x1 = df.loc[j, "Latitude"]
                y1 = df.loc[j, "Longitude"]
                x2 = df.loc[j + 1, "Latitude"]
                y2 = df.loc[j + 1, "Longitude"]
                distance = geodesic((x1, y1), (x2, y2)).m
                t1_str = df.loc[j, "Time"]
                d1_str = df.loc[j, "Date"]
                t2_str = df.loc[j + 1, "Time"]
                d2_str = df.loc[j + 1, "Date"]
                t1 = datetime.datetime(*map(int, d1_str.split('-')), *map(int, t1_str.split(':')))
                t2 = datetime.datetime(*map(int, d2_str.split('-')), *map(int, t2_str.split(':')))
                duration = (t2 - t1).seconds
                if duration == 0:
                    # 直接舍弃有瞬移的轨迹
                    flag = False
                    break
                X[j] = distance / duration
        if len(df)<max_traj_len:
            l = len(df)
            for j in range(l-1):
                #计算平均速度
                x1 = df.loc[j,"Latitude"]
                y1 = df.loc[j,"Longitude"]
                x2 = df.loc[j+1,"Latitude"]
                y2 = df.loc[j+1,"Longitude"]
                distance = geodesic((x1,y1),(x2,y2)).m
                t1_str = df.loc[j,"Time"]
                d1_str = df.loc[j,"Date"]
                t2_str = df.loc[j+1,"Time"]
                d2_str = df.loc[j+1,"Date"]
                t1 = datetime.datetime(*map(int,d1_str.split('-')),*map(int,t1_str.split(':')))
                t2 = datetime.datetime(*map(int,d2_str.split('-')),*map(int, t2_str.split(':')))
                duration = (t2-t1).seconds
                if duration == 0:
                    #直接舍弃有瞬移的轨迹
                    flag = False
                    break
                X[j] = distance/duration
            #不足的部分使用全程的平均速度补齐
            x1 = df.loc[0,"Latitude"]
            y1 = df.loc[0, "Longitude"]
            x2 = df.loc[l-1, "Latitude"]
            y2 = df.loc[l-1, "Longitude"]
            distance = geodesic((x1, y1), (x2, y2)).m
            t1_str = df.loc[0, "Time"]
            d1_str = df.loc[0, "Date"]
            t2_str = df.loc[l-1, "Time"]
            d2_str = df.loc[l-1, "Date"]
            t1 = datetime.datetime(*map(int, d1_str.split('-')), *map(int, t1_str.split(':')))
            t2 = datetime.datetime(*map(int, d2_str.split('-')), *map(int, t2_str.split(':')))
            duration = (t2 - t1).seconds
            avg_v = distance/duration
            X[l:] = avg_v
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


