'''
Created on 13 Dec 2021

@author: aftab
'''
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging





class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid =32
        self.in_head = 1
        self.out_head = 1
        
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
       
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(x, dim=1)
    
    
def callGAT(dat):
    stats={}
    global dataset
    dataset = dat
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    
    model = GAT().to(device)
    data = dataset.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_data=[]
    model.train()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss_data.append(loss)
        loss.backward()
        optimizer.step()
        
    model.eval()
    _, pred = model(data).max(dim=1)
    
    correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / data.val_mask.sum().item()
    #print('GAT Accuracy Validation: {:.4f}'.format(acc))
    stats['val_acc']=acc
    
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    #print('GAT Accuracy Test: {:.4f}'.format(acc))
    
    stats['test_acc']=acc
    return stats,loss_data
    