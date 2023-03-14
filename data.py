import pandas as pd
import os

#取出不含label的用户轨迹
#保存格式一条轨迹一个文件，一行一个点，加上label和user_id标签
source_data_path = 'source/Geolife Trajectories 1.3/Data/'
total_traj_id = 0
for userid in range(182):
    label_path = source_data_path+'{:0>3d}/labels.txt'.format(userid)
    traj_path = source_data_path+'{:0>3d}/Trajectory/'.format(userid)
    if os.path.exists(label_path):
        label_df = pd.read_table(label_path,sep = '\t')
        traj_df = pd.DataFrame(columns=["Latitude","Longitude","0","Altitude","Date_day","Date","Time"])
        for root,dirs,files in os.walk(traj_path):
            for f in files:
                traj_df = pd.concat([traj_df,pd.read_csv(traj_path+f,skiprows=6,names=["Latitude","Longitude","0","Altitude","Date_day","Date","Time"])],ignore_index=True)
        #匹配对应的此用户的轨迹
        for i in range(len(label_df)):
            mode = label_df.loc[i,"Transportation Mode"]
            start_date = label_df.loc[i,"Start Time"].split(" ")[0].replace("/","-")
            start_time = label_df.loc[i,"Start Time"].split(" ")[1]
            end_date = label_df.loc[i,"End Time"].split(" ")[0].replace("/","-")
            end_time = label_df.loc[i,"End Time"].split(" ")[1]
            start_node = traj_df[(traj_df["Date"]==start_date)&(traj_df["Time"]==start_time)]
            end_node = traj_df[(traj_df["Date"]==end_date)&(traj_df["Time"]==end_time)]
            if len(start_node) > 0 and len(end_node)>0:
                start_index = start_node.index.asi8[0]
                end_index = end_node.index.asi8[0]
                result_df = pd.DataFrame(columns=["Latitude","Longitude","Altitude","Date","Time","user_id","mode"])
                for j in range(start_index,end_index+1):
                    result_df = result_df.append({"Latitude":traj_df.loc[j,"Latitude"],"Longitude":traj_df.loc[j,"Longitude"],"Altitude":traj_df.loc[j,"Altitude"],"Date":traj_df.loc[j,"Date"],"Time":traj_df.loc[j,"Time"],"user_id":userid,"mode":mode},ignore_index=True)
                result_df.to_csv('traj/{}.csv'.format(total_traj_id))
                total_traj_id = total_traj_id + 1


