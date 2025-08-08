import numpy as np




def EIMA(df_hyper_matrix, edge_weights, seeds, beta, knum, infect_matrix):
    NNum, ENum = df_hyper_matrix.shape
    nodes_prob = []
    nodes_data = np.zeros(NNum)

    for i in seeds:
        nodes_data[i] = 1
    nodes_prob.append(nodes_data.copy())

    for kk in range(knum):
        nodes_data = nodes_prob_update(infect_matrix, nodes_data)
        nodes_prob.append(nodes_data.copy())
    nodes_prob_k = nodes_prob[-1]

    return nodes_prob_k


def cal_infect_matrix(df_hyper_matrix, edge_weights, beta):
    NNum, ENum = df_hyper_matrix.shape
    infect_matrix = np.zeros((NNum, NNum))

    for i in range(NNum):
        Edge_index = np.where(df_hyper_matrix.values[i, :] == 1)[0]
        values_list = edge_weights[Edge_index]
        total_sum = sum(values_list)
        values_normal = [value / total_sum for value in values_list]

        for j in range(NNum):
            if i != j:
                summ = 1
                for r,e in enumerate(Edge_index):
                    if df_hyper_matrix.values[j, e] == 1:
                        summ = summ * (1 - values_normal[r] * beta)
                infect_matrix[i,j] = 1 - summ
    return infect_matrix


def nodes_prob_update(infect_matrix, nodes_data):
    raw_nodes_data = nodes_data.copy()
    for i in range(len(nodes_data)):
        raw_prob = raw_nodes_data[i]
        neighbors = np.where(infect_matrix[i, :] > 0)[0]
        new_p_list_infect = raw_nodes_data[neighbors]
        new_p_list_spread = np.array([infect_matrix[x, i] for x in neighbors])
        new_p_list = list(new_p_list_infect * new_p_list_spread)
        nodes_data[i] = prob_meanwhile(raw_prob, new_p_list)
    return nodes_data


def prob_meanwhile(raw_prob, new_p_list):
    new_p_list.append(raw_prob)
    result = 1 - np.prod(1-np.array(new_p_list))
    return result




