'''
Created on 13 Dec 2021

@author: aftab
'''
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import logging




class GCN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        #self.conv2 = GCNConv(30, 10)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        #print(data)
        x, edge_index = data.x, data.edge_index
        #print('Node Matrix:',x)
        #print('Edge Matrix in COO format:',edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.6, training=self.training)
        #x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


def callGCN(data):
    stats= {}
    global dataset
    dataset= data
    #print(dataset)
    if dataset:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN().to(device)
        loss_data=[]
        data = dataset.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        logging.info('Start of Training')
        model.train()
        logging.info('End of Training')
        logging.info('Start of Epochs')
        #print(data.train_mask.items())
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss_data.append(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        logging.info('end of Epochs')
        
        pred = model(data).argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
        #print(f'GCN Accuracy validation: {acc:.4f}')
        stats['val_acc']=acc
       
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        #print(f'GCN Accuracy Test: {acc:.4f}')
        stats['test_acc']=acc
        return stats,loss_data
        
        
        
        