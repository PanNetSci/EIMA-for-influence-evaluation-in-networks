from Dataloader import dataloader
from Algorithms import algorithms
import os
import numpy as np
import pandas as pd




if __name__ == '__main__':

    paths = os.listdir('./Datasets/')

    for i, path in enumerate(paths):
        if i % 2 == 0:
            path_data = './Datasets/' + path
            path_weight = './Datasets/' + path[:-4] + '_weights.txt'

            dl = dataloader(path_data, path_weight)
            dl.dataload()
            df_hyper_matrix = dl.hyper_matrix
            edge_weights = dl.edge_weights

            K = 30
            beta = 0.01
            mtkl = 100
            max_iteration = 10
            knum1 = 10
            knum2 = 10
            knum3 = 10


            seeds_list = []
            cost_time_list = []

            methods = ['EIMA_heuristic', 'EIMA_avoid', 'EIMA_greedy', 'HADP', 'H-RIS', 'H-CI(I=1)', 'H-Degree', 'Degree', 'General-greedy']

            seeds_list_EIMA_heuristic, cost_time_EIMA_heuristic = algorithms.EIMA_heuristic(df_hyper_matrix, edge_weights, beta, K, knum1)
            seeds_list_EIMA_avoid, cost_time_EIMA_avoid = algorithms.EIMA_avoid(df_hyper_matrix, edge_weights, beta, K, knum3)
            seeds_list_EIMA_greedy, cost_time_EIMA_greedy = algorithms.EIMA_greedy(df_hyper_matrix, edge_weights, beta, K, knum2)
            seeds_list_HADP       ,  cost_time_HADP         =  algorithms.HADP(df_hyper_matrix, K)
            seeds_list_RIS        ,  cost_time_RIS          =  algorithms.RIS(df_hyper_matrix, K, 0.01, 200)
            seeds_list_CI1        ,  cost_time_CI1          =  algorithms.CI(df_hyper_matrix, K, 1)
            seeds_list_HDegree    ,  cost_time_HDegree      =  algorithms.HDegree(df_hyper_matrix, K)
            seeds_list_Degree     ,  cost_time_Degree       =  algorithms.degreemax(df_hyper_matrix, K)
            seeds_list_Greedy, cost_time_Greedy = algorithms.generalGreedy(df_hyper_matrix, edge_weights, beta, max_iteration, K, mtkl)



            seeds_list.append(seeds_list_EIMA_heuristic)
            seeds_list.append(seeds_list_EIMA_avoid)
            seeds_list.append(seeds_list_EIMA_greedy)
            seeds_list.append(seeds_list_HADP)
            seeds_list.append(seeds_list_RIS)
            seeds_list.append(seeds_list_CI1)
            seeds_list.append(seeds_list_HDegree)
            seeds_list.append(seeds_list_Degree)
            seeds_list.append(seeds_list_Greedy)



            cost_time_list.append(cost_time_EIMA_heuristic)
            cost_time_list.append(cost_time_EIMA_avoid)
            cost_time_list.append(cost_time_EIMA_greedy)
            cost_time_list.append(cost_time_HADP)
            cost_time_list.append(cost_time_RIS)
            cost_time_list.append(cost_time_CI1)
            cost_time_list.append(cost_time_HDegree)
            cost_time_list.append(cost_time_Degree)
            cost_time_list.append(cost_time_Greedy)




            seeds_result = pd.DataFrame(seeds_list).T
            seeds_result.loc[len(seeds_result.index), :] = np.array(cost_time_list)
            seeds_result.index = list(np.arange(1,K+1)) + ['Time_Cost']
            seeds_result.columns = methods
            seeds_result.to_excel('./seeds_result/' + path[:-4] + '.xlsx', sheet_name="Seeds_List")




    
