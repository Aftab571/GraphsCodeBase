import pandas as pd
from torch.nn import ReLU
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
from tqdm import tqdm
import time
import os.path as osp
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from sklearn import preprocessing


torch.cuda.empty_cache()

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def get_edges(query):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456"))
    with driver.session() as session:
        result = session.run(query)
    return result

def getData(query):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456"))
    with driver.session() as session:
        result = session.run(query)
        #return result
        return pd.DataFrame([r.values() for r in result], columns=result.keys())



def fetch_node_features():
    patient_nodes_query = """
        MATCH(p:Patient)  RETURN  p.race as race, p.ethnicity as ethnicity, p.gender as gender, p.marital as marital
        """
    encounter_nodes_query = """
        MATCH(e:Encounter)  RETURN e.description as desc
        """
    condition_nodes_query = """
        MATCH(c:Condition) with distinct c.code as code,c RETURN code
        """
    drug_nodes_query = """
        MATCH(d:Drug) with distinct d.code as code,d RETURN code
        """

    pat_df = getData(patient_nodes_query)
    enc_df = getData(encounter_nodes_query)
    con_df = getData(condition_nodes_query)
    drg_df = getData(drug_nodes_query)

    return pat_df, enc_df, con_df, drg_df

def fetch_cites_edge_list():
    patient_hasEnc_encounter_query = """
     MATCH (n:Patient)-[:HAS_ENCOUNTER]->(m: Encounter) RETURN ID(n), ID(m)
    """
    encounter_hasCond_condition_query = """
     MATCH (n:Encounter)-[:HAS_CONDITION]->(m: Condition) RETURN ID(n), ID(m)
    """
    encounter_hasdrg_drug_query = """
     MATCH (n:Encounter)-[:HAS_DRUG]->(m: Drug) RETURN ID(n), ID(m)
    """


    pat_HE_enc = getData(patient_hasEnc_encounter_query)
    enc_HC_con = getData(encounter_hasCond_condition_query)
    enc_HD_drg = getData(encounter_hasdrg_drug_query)

    return pat_HE_enc,enc_HC_con,enc_HD_drg

def preprocOneHot(df):
    One_enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(One_enc.fit_transform(df).toarray())
    return enc_df

def UMLbert(df):
    model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
    emd_lst=[]
    df= df.head(5)
    for i,x in df.iterrows():
        #print(x)
        inputs_1 = tokenizer(x['desc'], return_tensors='pt')
        sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
        emd_lst.append(sent_1_embed)
    return emd_lst

def reverse_edge_index(data):
    cols = data.columns.tolist()
    rev_cols = reversed(cols)
    df1=data.reindex(columns=rev_cols)
    return df1.values.tolist()

def create_train_test_mask(y):
    print(y)
    x_train, x_test = train_test_split(pd.Series(y), test_size=0.3, random_state=42)
    train_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    test_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    train_mask[x_train.index] = True 
    test_mask[x_test.index] = True
    return train_mask, test_mask 
 
def main():
 
    p,e,c,d = fetch_node_features()
    
    p_HE_e, e_HC_c, e_HD_d = fetch_cites_edge_list()

    print(preprocOneHot(e))
    print(preprocOneHot(c))
    print(preprocOneHot(d))
   
    data = HeteroData()

    st_time_nodes = time.time()
    data['patient'].x = torch.tensor(preprocOneHot(p).values, dtype = torch.float)
    le = preprocessing.LabelEncoder()
    le.fit(e['desc'].values)
    encList= le.transform(e['desc'].values.tolist())
    data['encounter'].x =  torch.tensor(encList, dtype = torch.float)
    data['condition'].x = torch.tensor(preprocOneHot(c).values.tolist(), dtype = torch.float)
    data['drug'].x = torch.tensor(preprocOneHot(d).values.tolist(), dtype = torch.float)
    data['patient', 'hasEncounter', 'encounter'].edge_index = torch.tensor(p_HE_e.values.tolist(), dtype=torch.long).t().contiguous() 
    data['encounter', 'hasCondition', 'condition'].edge_index = torch.tensor(e_HC_c.values.tolist(), dtype=torch.long).t().contiguous() 
    data['encounter', 'hasDrug', 'drug'].edge_index = torch.tensor(e_HD_d.values.tolist(), dtype=torch.long).t().contiguous() 
    data['encounter', 'rev_hasEncounter', 'patient'].edge_index = torch.tensor(reverse_edge_index(p_HE_e), dtype=torch.long).t().contiguous() 
    data['condition', 'rev_hasCondition', 'encounter'].edge_index = torch.tensor(reverse_edge_index(e_HC_c), dtype=torch.long).t().contiguous()
    data['drug', 'rev_hasDrug', 'encounter'].edge_index = torch.tensor(reverse_edge_index(e_HD_d), dtype=torch.long).t().contiguous() 
    le = preprocessing.LabelEncoder()
    le.fit(c['code'].values)
    labelList= le.transform(c['code'].values.tolist())
    print(labelList)
    data['condition'].y =  torch.tensor(labelList, dtype = torch.long)

    train, test = create_train_test_mask(labelList)
    print(train)
    print(test)

    data['condition'].train_mask = train
    data['condition'].test_mask =  test
    data.num_classes=len(labelList)
    



    
    train_input_nodes = ('condition', data['condition'].train_mask)
    val_input_nodes = ('condition', data['condition'].test_mask)
    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    print(data)

    train_loader = NeighborLoader(data, num_neighbors=[2] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[2] * 2,
                                input_nodes=val_input_nodes, **kwargs)
    print("hello")
    print(train_loader)
    print(val_loader)

    # #data.transform = T.ToUndirected(merge=True)

    model = Sequential('x, edge_index', [
        (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (Linear(-1, data.num_classes), 'x -> x'),
    ])
    model = to_hetero(model, data.metadata(), aggr='sum').to(device)


    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device, 'edge_index')
        model(batch.x_dict, batch.edge_index_dict)


    def train():
        model.train()

        total_examples = total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size

        return total_loss / total_examples


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

        return total_correct / total_examples


    init_params()  # Initialize parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 2):
        loss = train()
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

    #Model evaluation
    # model.eval()
    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # print(f'Accuracy: {acc:.4f}')
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)['paper']
    pred = out.argmax(dim=-1)
    mask = data['paper']['test_mask']
    acc = (pred[mask] == data['paper'].y[mask]).sum() / mask.sum()
    print(f'Accuracy: {acc:.4f}')
       
    



if __name__ == "__main__":
   
    main()