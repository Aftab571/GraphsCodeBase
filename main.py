import yaml
import connection as conn
import datapreprocessing as prep
import Visualize as vis
import GCNTest as gcn
import GATTest as gat
import GSageTest as gsage

def load_config():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)
    return cfg

def Average(lst):
    return sum(lst) / len(lst)
    
def main():
    gcn_list=[]
    gat_list=[]
    gsage_list=[]
    iter_list=[]
    cfg= load_config()
    neoConn=conn.Connection(cfg)
    edge_query = str(cfg["queries"][0]['GET_PAPER_CITES_CQL'])
    node_query = str(cfg["queries"][1]['GET_ALL_NODES_CQL'])
    neoData_edge= neoConn.fetch_data(edge_query)
    neoData_node= neoConn.fetch_data(node_query)
    data, stats= prep.DataPrep.convertData(df1=neoData_edge,df2= neoData_node)
    print(stats)
    file = open("stats.txt","r+")
    file.truncate(0)
    file.close()
    with open('stats.txt', 'w') as f:
        f.write(str(stats)+"\n")
        f.close()
    for i in range(10):
        gcn_stats,gcn_loss = gcn.callGCN(data)
        gat_stats,gat_loss = gat.callGAT(data)
        gsage_stats = gsage.callGSage(data)
        
        gcn_list.append(gcn_stats['test_acc'])
        gat_list.append(gat_stats['test_acc'])
        gsage_list.append(gsage_stats['test_acc'])
        iter_list.append(i)
        with open('stats.txt', 'a') as f:
            iteration = "----------------Iteration"+str(i)+"----------------------\n"
            gcn_stats_str = "GCN Stats:"+str(gcn_stats)+"\n"
            gat_stats_str = "GAT Stats:"+str(gat_stats)+"\n"
            gsage_stats_str = "GSage Stats:"+str(gsage_stats)+"\n"
            
            f.write(iteration)
            f.write(gcn_stats_str)
            f.write(gat_stats_str)
            f.write(gsage_stats_str)
            f.close()
        print("----------------Iteration",str(i),"----------------------")
        print("GCN Stats:",str(gcn_stats))
        print("GAT Stats:",str(gat_stats))
        print("GSage Stats:",str(gsage_stats))
        vis.plot_train_loss(gcn_loss,gat_loss,i)
        
    with open('stats.txt', 'a') as f:
        avg_gcn= "Average Test Accuracy for GCN: "+str(Average(gcn_list))+"\n"
        avg_gat= "Average Test Accuracy for GAT: "+str(Average(gat_list))+"\n"
        avg_gsage= "Average Test Accuracy for GSAGE: "+str(Average(gsage_list))+"\n"
        f.write(avg_gcn)
        f.write(avg_gat)
        f.write(avg_gsage)
        f.close()
    
    print("Average Test Accuracy for GCN: ", str(Average(gcn_list)))
    print("Average Test Accuracy for GAT: ", str(Average(gat_list)))
    print("Average Test Accuracy for GSAGE: ", str(Average(gsage_list)))
        
    vis.vis_graph_Models(gcn_list,gat_list,gsage_list,iter_list)
   

    

if __name__== "__main__":
   main()



