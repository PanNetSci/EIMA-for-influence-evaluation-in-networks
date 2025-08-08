import numpy as np


class Adaptive_Dissemination:

    def AD(hyper_matrix, edge_weights, seeds_list, max_iteration, beta):
        I_node_list = [list(seeds_list)]
        I_Nodes = seeds_list.copy()
        inf_information = []

        for t in range(max_iteration):
            infected_nodes_t = []
            for inode in I_Nodes:
                inode_edges = Edges_of_nodes([inode], hyper_matrix)
                values_list = edge_weights[inode_edges]
                total_sum = sum(values_list)
                inode_edges_values_normal = [value/total_sum for value in values_list]
                inode_edges_dic = dict( zip(inode_edges, inode_edges_values_normal) )

                for edge in inode_edges:
                    spread_pro = inode_edges_dic[edge]
                    inf_nodes_of_edge = Spreading_edge(edge, spread_pro, hyper_matrix, beta)
                    unique_nodes = [x for x in inf_nodes_of_edge if x not in I_Nodes and x not in infected_nodes_t]
                    inf_information.extend([[inode, edge, x] for x in unique_nodes])
                    infected_nodes_t.extend(unique_nodes)
            I_Nodes.extend(infected_nodes_t)
            I_node_list.append(I_Nodes.copy())

        I_edge_list = [Edges_of_nodes(x, hyper_matrix) for x in I_node_list]
        I_node_scale = [len(x) for x in I_node_list]
        I_edge_scale = [len(x) for x in I_edge_list]

        return (I_node_list, I_node_scale, I_edge_list, I_edge_scale, inf_information)





def MinMaxScaler(array, scale_min, scale_max):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val) * (scale_max - scale_min) + scale_min
    return scaled_array



def Edges_of_nodes(nodes, hyper_matrix):
    hyper_matrix = hyper_matrix.values
    edges = []
    for inode in nodes:
        inode_edges = np.where(hyper_matrix[inode, :] == 1)[0]
        unique_edges = [x for x in inode_edges if x not in edges]
        edges.extend(unique_edges)
    return edges



def Nodes_of_edges(edges, hyper_matrix):
    hyper_matrix = hyper_matrix.values
    nodes = []
    for iedge in edges:
        iedge_nodes = np.where(hyper_matrix[:, iedge] == 1)[0]
        unique_nodes = [x for x in iedge_nodes if x not in nodes]
        nodes.extend(unique_nodes)
    return nodes



def Spreading_edge(edge, spread_pro, hyper_matrix, beta):
    infected_nodes = []
    involved_nodes = np.array(Nodes_of_edges([edge], hyper_matrix))

    if np.random.random() <= spread_pro:
        pro_list = np.random.random(len(involved_nodes))
        infected_nodes = involved_nodes[np.where(pro_list <= beta)[0]]
    return infected_nodes



def Neighbors_of_node(node, hyper_matrix):
    all_edges = Edges_of_nodes(node, hyper_matrix)
    all_nodes = []
    for i in all_edges:
        nodess = Nodes_of_edges([i], hyper_matrix)
        nodes = []
        for x in nodess:
            if x not in all_nodes:
                if x not in node:
                    nodes.append(x)
        all_nodes.extend(nodes)
    return all_nodes



