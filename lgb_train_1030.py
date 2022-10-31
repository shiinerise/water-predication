# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import tqdm
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from itertools import repeat
from sklearn.preprocessing import RobustScaler, normalize
from sklearn import metrics
from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
import gc
from itertools import repeat
import warnings
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
warnings.filterwarnings("ignore")

def build_features1(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['yb_id'] = df['year'].astype(str) + df['month'].astype(str) + df['day'].astype(str)
    df['JS_NH3_CS'] = df['JS_NH3'] - df['CS_NH3']
    df['JS_TN_CS'] = df['JS_TN'] - df['CS_TN']
    df['JS_LL_CS'] = df['JS_LL'] -df['CS_LL']
    df['JS_COD_CS'] = df['JS_COD'] - df['CS_COD']
    df['JS_SW_CS'] = df['JS_SW'] - df['CS_SW']
    targets = df.loc[:, 'Label1']
    print('target isnull: ')
    print(targets.isnull().sum())
    # targets = targets.to_numpy().reshape(-1, 1)
    train_df = df.drop(['time', 'N_HYC_NH4', 'N_HYC_XD', 'N_HYC_MLSS', 'N_HYC_JS_DO', 'N_HYC_DO', 'N_CS_MQ_SSLL', 'N_QY_ORP', 'Label1', 'Label2'], axis=1)
    print(train_df.head())
    RS = RobustScaler()  # 归一化，这里没有采用maxmin归一化方式
    train_df = RS.fit_transform(train_df)
    # train_df = train.reshape(-1, 1, train.shape[-1])
    return train_df, targets

def build_features2(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['yb_id'] = df['year'].astype(str) + df['month'].astype(str) + df['day'].astype(str)
    df['JS_NH3_CS'] = df['JS_NH3'] - df['CS_NH3']
    df['JS_TN_CS'] = df['JS_TN'] - df['CS_TN']
    df['JS_LL_CS'] = df['JS_LL'] -df['CS_LL']
    df['JS_COD_CS'] = df['JS_COD'] - df['CS_COD']
    df['JS_SW_CS'] = df['JS_SW'] - df['CS_SW']
    targets = df.loc[:, 'Label2']
    print('target isnull: ')
    print(targets.isnull().sum())
    # targets = targets.to_numpy().reshape(-1, 1)
    train_df = df.drop(['time', 'B_HYC_NH4', 'B_HYC_XD', 'B_HYC_MLSS', 'B_HYC_JS_DO', 'B_HYC_DO', 'B_CS_MQ_SSLL', 'B_QY_ORP', 'Label1', 'Label2'], axis=1)
    print(train_df.head())
    RS = RobustScaler()  # 归一化，这里没有采用maxmin归一化方式
    train_df = RS.fit_transform(train_df)
    # train_df = train.reshape(-1, 1, train.shape[-1])
    return train_df, targets

def build_features3(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['yb_id'] = df['year'].astype(str) + df['month'].astype(str) + df['day'].astype(str)
    df['JS_NH3_CS'] = df['JS_NH3'] - df['CS_NH3']
    df['JS_TN_CS'] = df['JS_TN'] - df['CS_TN']
    df['JS_LL_CS'] = df['JS_LL'] -df['CS_LL']
    df['JS_COD_CS'] = df['JS_COD'] - df['CS_COD']
    df['JS_SW_CS'] = df['JS_SW'] - df['CS_SW']
    # targets = df.loc[:, 'Label1']
    # print('target isnull: ')
    # print(targets.isnull().sum())
    # targets = targets.to_numpy().reshape(-1, 1)
    train_df = df.drop(['time', 'N_HYC_NH4', 'N_HYC_XD', 'N_HYC_MLSS', 'N_HYC_JS_DO', 'N_HYC_DO', 'N_CS_MQ_SSLL', 'N_QY_ORP'], axis=1)
    print(train_df.head())
    RS = RobustScaler()  # 归一化，这里没有采用maxmin归一化方式
    train_df = RS.fit_transform(train_df)
    # train_df = train.reshape(-1, 1, train.shape[-1])
    return train_df

def build_features4(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['yb_id'] = df['year'].astype(str) + df['month'].astype(str) + df['day'].astype(str)
    df['JS_NH3_CS'] = df['JS_NH3'] - df['CS_NH3']
    df['JS_TN_CS'] = df['JS_TN'] - df['CS_TN']
    df['JS_LL_CS'] = df['JS_LL'] -df['CS_LL']
    df['JS_COD_CS'] = df['JS_COD'] - df['CS_COD']
    df['JS_SW_CS'] = df['JS_SW'] - df['CS_SW']
    # targets = df.loc[:, 'Label2']
    # print('target isnull: ')
    # print(targets.isnull().sum())
    # targets = targets.to_numpy().reshape(-1, 1)
    train_df = df.drop(['time', 'B_HYC_NH4', 'B_HYC_XD', 'B_HYC_MLSS', 'B_HYC_JS_DO', 'B_HYC_DO', 'B_CS_MQ_SSLL', 'B_QY_ORP'], axis=1)
    print(train_df.head())
    RS = RobustScaler()  # 归一化，这里没有采用maxmin归一化方式
    train_df = RS.fit_transform(train_df)
    # train_df = train.reshape(-1, 1, train.shape[-1])
    return train_df

if __name__ == "__main__":
    ####===============数据预处理和特征工程================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_save_dir = './save_weights/'
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    df_train = pd.read_csv('./train_dataset.csv', encoding='gbk')
    df1 = df_train[df_train['Label1'].isnull().values == False]
    train_df1, targets1 = build_features1(df1)
    print('训练集的形状：', train_df1.shape)
    print('训练目标的形状：', targets1.shape)

    train_data1, val_data1, train_target1, val_target1 = train_test_split(train_df1, targets1, test_size=0.2,
                                                                      random_state=0)
    reg_lgb1 = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression',
    )

    # 训练模型
    reg_lgb1.fit(train_data1, train_target1)
    val_pred1 = reg_lgb1.predict(val_data1)
    score1 = np.sqrt(mean_squared_error(val_target1, val_pred1))
    print('LightGBM score ', score1)
    # LightGBM socre  0.156274722993209

    df2 = df_train[df_train['Label2'].isnull().values == False]
    train_df2, targets2 = build_features2(df2)
    print('训练集的形状：', train_df2.shape)
    print('训练目标的形状：', targets2.shape)

    train_data2, val_data2, train_target2, val_target2 = train_test_split(train_df2, targets2, test_size=0.2,
                                                                          random_state=0)
    reg_lgb2 = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression',
    )

    # 训练模型
    reg_lgb2.fit(train_data2, train_target2)
    val_pred2 = reg_lgb2.predict(val_data2)
    score2 = np.sqrt(mean_squared_error(val_target2, val_pred2))
    print('LightGBM score ', score2)
    # LightGBM socre  0.156274722993209

    loss_all = (score1 + score2) / 2
    score = (1/(1+loss_all))*1000
    print(score)

    test = pd.read_csv('evaluation_public.csv')
    test_df1 = build_features3(test)
    test_df2 = build_features4(test)
    test_target1 = reg_lgb1.predict(test_df1)
    test_target2 = reg_lgb2.predict(test_df2)
    submission = pd.read_csv('sample_submission.csv')
    submission['Label1'] = test_target1
    submission['Label2'] = test_target2
    submission.to_csv('submissionlgb_1030.csv', index=False)
    joblib.dump(reg_lgb1, 'save_weights/tree/lgb1_1030.pkl')
    joblib.dump(reg_lgb2, 'save_weights/tree/lgb2_1030.pkl')
    # joblib.dump(reg_lgb2, 'save_weights/tree/lgb2_1030_047.pkl')
    # joblib.load('loan_model.pkl')
