import torch
import re 
import numpy as np

def compute_mean_mad(dataloaders, label_property):
    for key in dataloaders:
        values = dataloaders[key].dataset.data[label_property]
        meann = torch.mean(values)
        ma = torch.abs(values - meann)
        mad = torch.mean(ma)
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars

def get_force_by_filename(path):

    start = 0
    pattern1 = "\[(.*?)\]\]"
    pattern2 = "\[(.*?)\]"
    pattern3 = "\[\[(.*?)\]"
    new_path = path.replace('.out.xyz','.xyz')
    a = []
    
    with open(new_path)as f:
        for line in f:
            if ']]' in line.upper():
                start = 0
                a_string = re.search(pattern1, line).group(1)
                a_list = a_string.split()
                map_object = map(float, a_list)
                a.append(list(map_object))
            elif start:
                a_string = re.search(pattern2, line).group(1)
                a_list = a_string.split()
                map_object = map(float, a_list)
                a.append(list(map_object))
            elif '[[' in line.upper():
                start = 1
                a_string = re.search(pattern3, line).group(1)
                a_list = a_string.split()
                map_object = map(float, a_list)
                a.append(list(map_object))
    return np.array(a)