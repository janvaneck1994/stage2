import os.path as osp
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from matplotlib import pyplot
import torch_optimizer as optim
import random
from models import GCNModel, LinearModel
from utils import *
import torch.nn.functional as F
import h5py
from torch_geometric.utils import structured_negative_sampling

def get_val_test(cv_path):
    r"""create loader for the validation and test set
    """
    params = {'batch_size': 100000,
                  'shuffle': False,
                  'num_workers': 6,
                  'drop_last' : False}

    df_val = pd.read_csv(cv_path+'val.csv')
    validation_set = PPI(df_val)
    val_loader = data.DataLoader(validation_set, **params)

    df_test = pd.read_csv(cv_path+'test.csv')
    test_set = PPI(df_test)
    test_loader = data.DataLoader(test_set, **params)

    return val_loader, test_loader

def get_link_labels(pos_edge_index, neg_edge_index):
    r"""create labels for the edges
    """
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def print_results(train_edges, loader, model):
    r"""print the results given the val or test loader
    """
    _, y_label, y_pred, loss = test(train_edges, loader, model)
    roc, prc = roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred)

    print('loss_val: {:.4f}'.format(loss.item()),
          'auroc_val: {:.4f}'.format(roc),
          'auprc_val: {:.4f}'.format(prc))

def save_result(train_edges, loader, model, score_output, model_output):
    r"""save the results of the val and test loader
    """
    edges, y_label, y_pred, loss_val = test(train_edges, loader, model)
    df = pd.DataFrame(edges)
    df['label'] = y_label
    df['score'] = y_pred

    if not os.path.exists(score_output):
        os.makedirs(score_output)
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    torch.save(model.state_dict(), model_output+'weights')
    df.to_csv(score_output+'results.csv', index=None)

def train(train_edges, model):
    r"""train the model
    """
    model.train()
    optimizer.zero_grad()

    train_edges_np = train_edges.cpu().numpy()
    train_nodes = np.unique(train_edges_np)
    train_to_idx, idx_to_train = get_node_mapping(train_nodes)

    #generate negatives with mapped positive edges
    #edges should be mapped to prevent allowing nodes not in training
    negatives = structured_negative_sampling(torch.tensor(train_to_idx(train_edges_np)), num_nodes=len(train_nodes))

    #keep one of the two positive nodes in the edges
    rand_node = random.randint(0, 1)
    u = negatives[rand_node]
    v = negatives[2]

    # backmap proteins
    negative_edge = torch.stack([u,v])
    negative_edge = idx_to_train(negative_edge.cpu().numpy())
    negative_edge = torch.tensor(negative_edge).to(device)

    total_edge = torch.cat([train_edges, negative_edge], dim=-1).type(torch.LongTensor).to(device)

    #create scores, labels, update parameters
    link_logits = model(features, train_edges, total_edge)
    link_labels = get_link_labels(train_edges, negative_edge)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(train_edges, loader, model):
    r"""test the model
    """
    model.eval()

    with torch.no_grad():
        edges = np.zeros((0, 2)).astype('int')
        y_pred = []
        y_label = []

        for i, (label, inp, index) in enumerate(loader):

            label = label.to(device)

            #create scores
            inp = torch.stack(inp).type(torch.LongTensor).to(device)
            output = model(features, train_edges, inp)

            n = torch.squeeze(output)

            loss = F.binary_cross_entropy_with_logits(n, label.float())

            # labels and scores
            label_ids = label.to('cpu').numpy()
            inp_ids = inp.to('cpu').numpy().T
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            edges = np.concatenate([edges, inp_ids])

    return edges, y_label, y_pred, loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(123)
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed(123)

path = '../data/processed/'

# get nodes
df_prot_list = pd.read_csv(path+'protein_list.csv', header=None)
idx = df_prot_list[0].tolist()
num_nodes = len(df_prot_list)
num_features = 1024

#get embeddings
embedding_file = "../data/human_proteome_protBertBFD.h5"
features_seq_np = np.zeros((num_nodes, num_features))
with h5py.File(embedding_file, "r") as f:
    for i, x in enumerate(df_prot_list[0]):
        features_seq_np[i] = np.array(f[x])

#for every cross validation
for cv in range(10):
    for model_name in ['GNN_one-hot','GNN_ProtBert','FFNN']:
        print(cv, model_name)

        # undersampling ratios data
        cv_path_1 = path + 'cv/' + str(cv) + '/1/'
        cv_path_10 = path + 'cv/' + str(cv) + '/10/'
        cv_path_100 = path + 'cv/' + str(cv) + '/100/'

        #training data
        df_train = pd.read_csv(cv_path_1+'train.csv')
        train_edges = df_train[df_train.iloc[:,2] == 1].iloc[:,:2].values
        train_edges = torch.LongTensor(train_edges.T).type(torch.LongTensor)

        #create test and validation loaders
        val_loader_1, test_loader_1 = get_val_test(cv_path_1)
        val_loader_10, test_loader_10 = get_val_test(cv_path_10)
        val_loader_100, test_loader_100 = get_val_test(cv_path_100)

        #embeddings or one hot encoded features
        with torch.no_grad():
            features_seq = torch.FloatTensor(features_seq_np)
            features_eye = torch.FloatTensor(np.eye(num_nodes))

        # set features and epochs for each model
        if model_name == 'GNN_one-hot':
            epochs = 750
            features = features_eye
            model = GCNModel(features.shape[1]).to(device)
        elif model_name == 'FFNN':
            epochs = 750
            features = features_seq
            model = LinearModel(features.shape[1]).to(device)
        elif model_name == 'GNN_ProtBert':
            epochs = 750
            features = features_seq
            model = GCNModel(features.shape[1]).to(device)

        features = features.to(device)
        train_edges = train_edges.to(device)

        optimizer = optim.RAdam(model.parameters(),
                               lr=0.01)

        for epoch in range(1, epochs + 1):
            loss_train = train(train_edges, model)
            if epoch % 100 == 0:
                print('-------','epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train), '-------')
                print_results(train_edges, val_loader_1, model)
                print_results(train_edges, val_loader_10, model)

        #save model results and model parameters
        model_weights = '../data/model_weights/'
        model_result = '../data/results/'
        output_1 = 'cv/'+str(cv)+'/1/'+model_name+'/'
        output_10 = 'cv/'+str(cv)+'/10/'+model_name+'/'
        output_100 = 'cv/'+str(cv)+'/100/'+model_name+'/'
        save_result(train_edges, test_loader_1, model, model_result+output_1, model_weights+output_1)
        save_result(train_edges, test_loader_10, model, model_result+output_10, model_weights+output_10)
        save_result(train_edges, test_loader_100, model, model_result+output_100, model_weights+output_100)
