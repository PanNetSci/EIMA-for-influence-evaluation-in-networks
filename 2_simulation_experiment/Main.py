from Dataloader import dataloader
from Simulation_Experiments import  simulation_experiments
import os
import pandas as pd


if __name__ == '__main__':
    paths = os.listdir('../1_search_seeds/Datasets/')

    for i, path in enumerate(paths):
        if i % 2 == 0:

            path_data = '../1_search_seeds/Datasets/' + path
            path_weight = '../1_search_seeds/Datasets/' + path[:-4] + '_weights.txt'

            dl = dataloader(path_data, path_weight)
            dl.dataload()
            df_hyper_matrix = dl.hyper_matrix
            edge_weights = dl.edge_weights

            R = 500
            beta = 0.005
            t = 45

            seeds_list = pd.read_excel('../1_search_seeds/seeds_result/' + path[:-4] + '.xlsx', sheet_name="Seeds_List", index_col=0, skipfooter=1)
            simulation_experiments.conduct_all_information(path, df_hyper_matrix, edge_weights, seeds_list, R, t, beta)



