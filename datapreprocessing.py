'''
Created on 14 Dec 2021

@author: aftab
'''
import torch
from torch_geometric.data import Data
import torch.utils.data as splitter
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing
from collections import Counter

class DataPrep:
    
    def convertData(df1, df2):
   
        x = torch.tensor(df2['features'].values.tolist(), dtype=torch.float)
        edge_index = torch.tensor(df1[['start','end']].values.tolist(), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        data.num_classes = len(df2['subject'].unique())
        le = preprocessing.LabelEncoder()
        le.fit(df2['subject'].values)
        labelList= le.transform(df2['subject'].values.tolist())
       
        
        X_train_complete, X_test, y_train, y_test = train_test_split(pd.Series(df2['features'].values.tolist()),df2['subject'].values.tolist(), test_size=0.20,random_state=1,stratify=df2['subject'].values.tolist())
        X_train, X_val, y_trainval, y_testval = train_test_split(X_train_complete,y_train, test_size=0.20,random_state=1,stratify=y_train)
    
        
        train_mask = torch.zeros(df2.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(df2.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(df2.shape[0], dtype=torch.bool)
      
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        val_mask[X_val.index] = True

        data.test_mask=test_mask
        data.train_mask=train_mask
        data.val_mask=val_mask

        data.y= torch.tensor(labelList).type(torch.LongTensor)
        
        dataDict= {
            "datasetSize": df2.shape[0],
            "train": X_train.shape[0],
            "val": X_val.shape[0],
            "test": X_test.shape[0],
            "trainL": Counter(y_trainval).values(),
            "valL": Counter(y_testval).values(),
            "testL": Counter(y_test).values()
            }        
      
        return data,dataDict