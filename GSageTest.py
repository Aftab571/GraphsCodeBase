'''
Created on 13 Dec 2021

@author: aftab
'''
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import logging


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim,normalize=True)
        self.conv2 = SAGEConv(hidden_dim, out_dim,normalize=True)
    
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        return torch.log_softmax(x, dim=-1)


def callGSage(data):
    stats= {}
    global dataset
    dataset= data
    hidden_dim = 32
    #print(dataset)
    if dataset:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GraphSAGE(in_dim=data.num_node_features, 
                 hidden_dim=hidden_dim, 
                 out_dim=dataset.num_classes).to(device)
       
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
        return stats
        
        
        
        