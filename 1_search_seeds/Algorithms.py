import numpy as np
import pandas as pd
import random
import copy
from Adaptive_Dissemination import Adaptive_Dissemination
import networkx as nx
from tqdm import tqdm
import time
from EIMA_Fitness import EIMA, cal_infect_matrix



class algorithms:


    def EIMA_heuristic(df_hyper_matrix, edge_weights, beta, k, knum):
        begin_time = time.time()
        node_fitness = []
        seed_list = []

        infect_matrix = cal_infect_matrix(df_hyper_matrix, edge_weights, beta)


        for i in tqdm(range(df_hyper_matrix.shape[0]), desc='EIMA_heuristic'):
            fitness = EIMA(df_hyper_matrix, edge_weights, [i], beta, knum, infect_matrix)
            node_fitness.append(fitness.sum())

        sorted_indices = sorted(range(len(node_fitness)), key=lambda ii: node_fitness[ii], reverse=True)

        for kk in range(k):
            seed_list.append(sorted_indices[:kk+1])

        end_time = time.time()
        cost_time = end_time - begin_time

        return seed_list, cost_time




    def EIMA_greedy(df_hyper_matrix, edge_weights, beta, k, knum):
        begin_time = time.time()
        seed_list_EIMA = []
        num_nodes = df_hyper_matrix.shape[0]
        seeds_Greedy = []

        infect_matrix = cal_infect_matrix(df_hyper_matrix, edge_weights, beta)

        for i in tqdm(range(k),desc='EIMA_greedy'):
            maxNode = 0
            maxfitness = 0
            for inode in range(num_nodes):
                if inode not in seeds_Greedy:
                    seeds_Greedy.append(inode)
                    fitness = EIMA(df_hyper_matrix, edge_weights, seeds_Greedy, beta, knum, infect_matrix)
                    seeds_Greedy.remove(inode)
                    if fitness.sum() > maxfitness:
                        maxNode = inode
                        maxfitness = fitness.sum()
            seeds_Greedy.append(maxNode)
            seed_list_EIMA.append(seeds_Greedy.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_EIMA, cost_time




    def EIMA_avoid(df_hyper_matrix, edge_weights, beta, k, knum):
        begin_time = time.time()

        m, n = df_hyper_matrix.shape
        seeds_HP = []
        infect_matrix = cal_infect_matrix(df_hyper_matrix, edge_weights, beta)
        nodes_values = []
        seeds = []

        for i in tqdm(range(m), desc='EIMA_avoid'):
            fitness = EIMA(df_hyper_matrix, edge_weights, [i], beta, knum, infect_matrix)
            nodes_values.append(fitness)

        for kk in range(k):
            best_node = algorithms.get_best_node(nodes_values)
            seeds.append(best_node)
            seeds_HP.append(seeds.copy())
            nodes_values = algorithms.update_nodes_values(best_node, nodes_values)

        end_time = time.time()
        cost_time = end_time - begin_time

        return seeds_HP, cost_time


    def get_best_node(nodes_value):
        m = len(nodes_value)
        nodes_value = np.array(nodes_value)
        sorted_nodes_index = sorted(list(range(m)), key=lambda x: nodes_value[x].sum(), reverse=True)
        picked_nodes = sorted_nodes_index[0]
        return picked_nodes


    def update_nodes_values(best_node, nodes_values):
        m = len(nodes_values)
        best_node_values = nodes_values[best_node].copy()
        initial_nodes_values = nodes_values.copy()

        for i in range(m):
            if i != best_node:
                nodes_values[i] = (1 - best_node_values) * initial_nodes_values[i]
            else:
                nodes_values[i] = np.zeros(m)

        return nodes_values




    def degreemax(df_hyper_matrix, K):
        begin_time = time.time()
        seed_list_degreemax = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for i in tqdm(range(0, K), desc='Degree finished'):
            seeds = algorithms.getSeeds_sta(degree, i)
            seed_list_degreemax.append(seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_degreemax, cost_time
       

    def getTotalAdj(df_hyper_matrix):

        deg_list = []
        N,M = df_hyper_matrix.shape
        nodes_arr = np.arange(N)
        for node in nodes_arr:
            node_list = []
            edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            node_set = np.unique(np.array(node_list))
            deg_list.append(len(list(node_set)) - 1)
        return np.array(deg_list)

    def getSeeds_sta(degree, i):
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
        degree_list = list(df_sort_matrix.loc['node_degree'])
        nodes_list = list(df_sort_matrix.loc['node_index'])
        chosed_arr = list(df_sort_matrix.loc['node_index'][:i])
        index = np.where(np.array(degree_list) == degree_list[i])[0]
        nodes_set = list(np.array(nodes_list)[index])
        while 1:
            node = random.sample(nodes_set, 1)[0]
            if node not in chosed_arr:
                chosed_arr.append(node)
                break
            else:
                nodes_set.remove(node)
                continue
        return chosed_arr


    def HDegree(df_hyper_matrix, K):
        begin_time = time.time()
        seed_list_HDegree = []
        degree = df_hyper_matrix.sum(axis=1)
        for i in tqdm(range(0, K), desc='H-Degree finished'):
            seeds = algorithms.getSeeds_sta(degree, i)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HDegree, cost_time


    def HADP(df_hyper_matrix, K):
        begin_time = time.time()
        seed_list_HUR = []
        seeds = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for j in tqdm(range(1, K+1), desc="HADP finished"):
            chosenNode = algorithms.getMaxDegreeNode(degree, seeds)
            seeds.append(chosenNode)
            seed_list_HUR.append(seeds.copy())
            algorithms.updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HUR, cost_time

    def getMaxDegreeNode(degree, seeds):
        degree_copy = copy.deepcopy(degree)
        global chosedNode
        while 1:
            flag = 0
            degree_matrix = algorithms.getDegreeList(degree_copy)
            node_index = degree_matrix.loc['node_index']
            for node in node_index:
                if node not in seeds:
                    chosedNode = node
                    flag = 1
                    break
            if flag == 1:
                break
        return chosedNode

    def getDegreeList(degree):
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)

    def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):
        edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]
        adj_set = []
        for edge in edge_set:
            adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        adj_set_unique = np.unique(np.array(adj_set))
        for adj in adj_set_unique:
            adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]
            adj_adj_set = []
            for each in adj_edge_set:
                adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
            if adj in adj_adj_set:
                adj_adj_set.remove(adj)
            sum = 0
            for adj_adj in adj_adj_set:
                if adj_adj in seeds:
                    sum = sum + 1
            degree[adj] = degree[adj] - sum
            


    def generalGreedy(df_hyper_matrix, edge_weights, beta, max_iteration, K, mtkl):
        begin_time = time.time()
        degree = df_hyper_matrix.sum(axis=1)
        seed_list_Greedy = []
        seeds = []

        for i in tqdm(range(0, K), desc="General-greedy finished"):
            scale_list_temp = []
            maxNode = 0
            maxScale = 0
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale_avg = []
                    for i in range(mtkl):
                        _, scale_temp, _, _, _ = Adaptive_Dissemination.AD(df_hyper_matrix, edge_weights, seeds, max_iteration, beta)
                        scale_avg.append(scale_temp[-1])
                    scale = np.array(scale_avg).mean()
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            seeds.append(maxNode)
            seed_list_Greedy.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_Greedy, cost_time


    def CI(df_hyper_matrix, K, l):
        begin_time = time.time()
        seed_list_CI = []
        seeds = []
        N, M = df_hyper_matrix.shape
        n = np.ones(N)
        CI_list = algorithms.computeCI(df_hyper_matrix, l)
        CI_arr = np.array(CI_list)
        for j in range(0, K):
            CI_chosed_val = CI_arr[np.where(n == 1)[0]]
            CI_chosed_index = np.where(n == 1)[0]
            index = np.where(CI_chosed_val == np.max(CI_chosed_val))[0][0]
            node = CI_chosed_index[index]
            n[node] = 0
            seeds.append(node)
            seed_list_CI.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_CI, cost_time

    def computeCI(df_hyper_matrix, l):
        CI_list = []
        degree = df_hyper_matrix.sum(axis=1)
        N,M = df_hyper_matrix.shape
        for i in tqdm(range(0, N), desc = "CI (l=%d) finished"%l):
            edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]
            if l == 1:
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set = np.unique(np.array(node_list))
            elif l == 2:
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set1 = np.unique(np.array(node_list))
                node_list2 = []
                edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
                edge_matrix[np.eye(M, dtype=np.bool_)] = 0
                df_edge_matrix = pd.DataFrame(edge_matrix)
                adj_edge_list = []
                for edge in edge_set:
                    adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
                adj_edge_set = np.unique(np.array(adj_edge_list))
                for each in adj_edge_set:
                    node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
                node_set2 = list(np.unique(np.array(node_list2)))
                for node in node_set2:
                    if node in list(node_set1):
                        node_set2.remove(node)
                node_set = np.array(node_set2)
            ki = degree[i]
            sum = 0
            for u in node_set:
                sum = sum + (degree[u] - 1)
            CI_i = (ki - 1) * sum
            CI_list.append(CI_i)
        return CI_list


    def RIS(df_hyper_matrix, K, lamda, theta):
        begin_time = time.time()
        seed_list_RIS = []
        S = []
        U = []
        N, M = df_hyper_matrix.shape

        for theta_iter in tqdm(range(0, theta), desc = "RIS finished"):
            df_matrix = copy.deepcopy(df_hyper_matrix)

            selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))), 1)[0]

            all_edges = np.arange(len(df_hyper_matrix.columns.values))
            prob = np.random.random(len(all_edges))
            index = np.where(prob > lamda)[0]
            for edge in index:
                df_matrix[edge] = 0

            adj_matrix = np.dot(df_matrix, df_matrix.T)
            adj_matrix[np.eye(N, dtype=np.bool_)] = 0
            df_adj_matrix = pd.DataFrame(adj_matrix)
            df_adj_matrix[df_adj_matrix > 0] = 1
            G = nx.from_numpy_array(df_adj_matrix.values)
            shortest_path = nx.shortest_path(G, target=selected_node)
            RR = []
            for each in shortest_path:
                RR.append(each)
            U.append(list(np.unique(np.array(RR))))

        for k in range(0, K):
            U_list = []
            for each in U:
                U_list.extend(each)
            dict = {}
            for each in U_list:
                if each in dict.keys():
                    dict[each] = dict[each] + 1
                else:
                    dict[each] = 1
            candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)
            chosed_node = candidate_list[0][0]
            S.append(chosed_node)
            seed_list_RIS.append(S.copy())
            for each in U:
                if chosed_node in each:
                    U.remove(each)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_RIS, cost_time







