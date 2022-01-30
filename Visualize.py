import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vis_graph_Models(gcn_list, gat_list,gsage_list,iter_list):

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    GCN= gcn_list
    GAT = gat_list
    GSAGE = gsage_list

    # Set position of bar on X axis
    br1 = np.arange(len(GCN))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, GCN, color ='blue', width = barWidth,
            edgecolor ='grey', label ='GCN')
    plt.bar(br2, GAT, color ='black', width = barWidth,
            edgecolor ='grey', label ='GAT')
    plt.bar(br3, GSAGE, color ='red', width = barWidth,
            edgecolor ='grey', label ='GSAGE')
    
    
    # Adding Xticks
    plt.xlabel('Iterations', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(GCN))],
            iter_list)
    
    plt.legend()
    plt.savefig("figures/gcnVSgatVSsage.png")
    #plt.show()
    
def plot_train_loss(gat_loss, gcn_loss, iter):
    #Visualising in Loss
    gat_losses_float = [float(loss.cpu().detach().numpy()) for loss in gat_loss]
    gat_losses_indices = [i for i, l in enumerate(gat_losses_float)]

    gcn_losses_float = [float(loss.cpu().detach().numpy()) for loss in gcn_loss]
    gcn_losses_indices = [i for i, l in enumerate(gcn_losses_float)]

    sns.lineplot(gat_losses_indices, gat_losses_float, x="epochs",y="loss")
    sns.lineplot(gcn_losses_indices,gcn_losses_float, x="epochs",y="loss")
    plt.legend(labels=["GAT","GCN"])
    name = "figures/train_losses"+str(iter)+".png"
    plt.savefig(name)
    #plt.show()
