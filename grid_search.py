# -*- coding: utf-8 -*-
#搜索最优的S,K值

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# dataset
import torch
import torch.nn as nn
import torch.utils.data as loader
from torch.utils.data import Dataset,DataLoader

def load_data_from_df(df,K =10,S=50,use_arma_mode = False ):
    '''加载xls 数据 ，并转换为输入特征向量
    df : data frame 格式
    use_arma_mode :  arma =True  模式的话，将返回 入库流量前K时刻值作为特征(无时间性)
    False :仅仅返回其它特征,返回为时间流数据, for lstm
    返回: numpy 格式数据
    '''
    datas = np.array(df)
    y = datas[:,0]
    x = datas[:,1:]


    if use_arma_mode :
        # 构建arna输入特征
        features = np.zeros( (len(y) - K-S , 50) ,dtype=np.float32)
        targets =np.zeros( (len(y) - K-S , 1) ,dtype=np.float32)
        for i in range(K+S, len(y)):
            f1 = x[i-K-S:i-S ,:]
            f1 = np.reshape(f1,(-1))
            f2 = y[i-K-S:i-S]
            f = np.concatenate((f1,f2))
            features[i-K-S,:] = f
            targets[i-K-S] = y[i]

        # 分配训练集和测试集
        permute = np.random.permutation(np.arange(0,len(targets)))
        features = features[permute]
        targets = targets[permute]

        f_train = features[0: int(0.9 * len(features)),: ]
        f_test = features[int(0.9 * len(features)):  ,: ]

        targets_train = targets[0: int(0.9 * len(features)) ] #
        targets_test = targets[int(0.9 * len(features)):]

    else :
        # f_train = x[0:int(0.9 * len(y)), :]
        # t_train = y[0:int(0.9 * len(y))]
        # f_test = x[int(0.9 * len(y)):, :]
        # t_test = y[int(0.9 * len(y)):]

        features = np.zeros((len(y) - K - S, K,4), dtype=np.float32)
        targets = np.zeros((len(y) - K - S,1), dtype=np.float32)
        for i in range(K+S, len(x)):
            f = x[i-K-S:i-S ,:]
            features[i-K-S,:] = f
            targets[i-K-S] = y[i]

        # 分配训练集和测试集
        permute = np.random.permutation(np.arange(0,len(targets)))
        features = features[permute]
        targets = targets[permute]

        f_train = features[0: int(0.9 * len(features)),: ]
        f_test = features[int(0.9 * len(features)):  ,: ]

        targets_train = targets[0: int(0.9 * len(features)) ] #
        targets_test = targets[int(0.9 * len(features)):]
    return f_train, targets_train, f_test, targets_test
def load_data(data,S,K):
    '''
    包装为 pytorch 数据
    data: numpy data
    S  时间偏移量
    K  多少个时间点前
    返回一个pytorch data loader
    '''


#todo: 归一化，预处理等等
class Dataset(Dataset):
    def __init__(self,x_train,y_train):
        self.x = x_train
        self.y = y_train
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return x_train[idx,:] ,y_train[idx]

class lstm(nn.Module):
    def __init__(self,input_dim,hidden_dim,batch_size,output_dim=1,num_layers=2):
        super(lstm,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)
        self.linear = nn.Linear(self.hidden_dim,1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        _, (hidden,_) = self.lstm(x)
        h_state = hidden[-1,:,:]
        out = self.linear(h_state)
        return out
# [-0.0576 0.0018 ..]



import matplotlib.pyplot as plt
if __name__ == '__main__':

    # 数据路径
    data_2015_spring = './data/2015年春.xls'
    data_2015_summer = './data/2015年夏.xls'
    data_2015_autumn = './data/2015年秋.xls'
    data_2015_winter = './data/2015年冬.xls'

    data_2016_spring = './data/2016年春.xls'
    data_2016_summer = './data/2016年夏.xls'
    data_2016_autumn = './data/2016年秋.xls'
    data_2016_winter = './data/2016年冬.xls'

    data_2017_spring = './data/2017年春.xls'
    data_2017_summer = './data/2017年夏.xls'
    data_2017_autumn = './data/2017年秋.xls'
    data_2017_winter = './data/2017年冬.xls'

    # 读取数据
    df_2015_spring = pd.read_excel(data_2015_spring, index_col='TIME')
    df_2015_summer = pd.read_excel(data_2015_summer, index_col='TIME')
    df_2015_autumn = pd.read_excel(data_2015_autumn, index_col='TIME')
    df_2015_winter = pd.read_excel(data_2015_winter, index_col='TIME')

    df_2016_spring = pd.read_excel(data_2016_spring, index_col='TIME')
    df_2016_summer = pd.read_excel(data_2016_summer, index_col='TIME')
    df_2016_autumn = pd.read_excel(data_2016_autumn, index_col='TIME')
    df_2016_winter = pd.read_excel(data_2016_winter, index_col='TIME')

    df_2017_spring = pd.read_excel(data_2017_spring, index_col='TIME')
    df_2017_summer = pd.read_excel(data_2017_summer, index_col='TIME')
    df_2017_autumn = pd.read_excel(data_2017_autumn, index_col='TIME')
    df_2017_winter = pd.read_excel(data_2017_winter, index_col='TIME')

    # 合并为一年
    df_2015 = pd.concat([df_2015_spring,df_2015_summer,df_2015_autumn,df_2015_winter])
    df_2016 = pd.concat([df_2016_spring,df_2016_summer,df_2016_autumn,df_2016_winter])
    df_2017 = pd.concat([df_2017_spring, df_2017_summer, df_2017_autumn, df_2017_winter])

    # grid search
    Ks = [10,20,30]
    Ss = [50,60,70,80]
    mses =[]
    KS_save = []
    for K in Ks :
        for S in Ss:
            x_train,y_train ,x_test, y_test = load_data_from_df(df_2015_spring,use_arma_mode=False,K=K,S=S)
            ds = Dataset(x_train,y_train)
            bs= 20
            epochs = 100
            da = DataLoader(ds, batch_size=bs, shuffle=True)
            lstm_model = lstm(4,100,bs,output_dim=1,num_layers=2) #input_dim,hidden_dim,batch_size,output_dim=1,num_layers=2
            loss_f = nn.MSELoss()
            optimiser = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)
            lstm_model.zero_grad()

            # 训练
            for e in range(epochs):
                total_loss = 0
                ind = 0
                for f, t in da:
                    pre = lstm_model(f)
                    loss = loss_f(pre,t)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total_loss+= loss.item()
                    ind +=1
                total_loss = total_loss /ind / bs
                if e%2 ==0 :
                    print("epoch {} :mse  {}".format(e, total_loss))

            #  测试
            preds = np.zeros(len(y_test),dtype=np.float)
            for i in range(len(y_test)):
                xx = x_test[i,:]
                f = torch.from_numpy(xx[np.newaxis,...])
                pre = lstm_model(f)
                pre = pre.detach().numpy()[0][0]
                preds[i] = pre

            # 计算mse
            mse  = np.mean((y_test[:, 0] - preds) ** 2)
            rmse = np.sqrt(np.mean((y_test[:,0] - preds)**2))
            mses.append(mse)
            KS_save.append([K,S])
    best_mse = np.min(np.arange(mses))

    print()
