from tqdm import tqdm
import os
from Adaptive_Dissemination import Adaptive_Dissemination



class simulation_experiments:

    def conduct_all_information(path, df_hyper_matrix, edge_weights, seeds_list, R, t, beta):
        if not os.path.exists('./beta = %s/'%beta + path[:-4]):
            os.makedirs('./beta = %s/'%beta + path[:-4])
        else:
            pass
        
        work_path = './beta = %s/%s/'%(beta, path[:-4])

        M, N = seeds_list.shape
        for col in range(N):
            for row in tqdm(range(M), desc="method %s: Loading..."%(seeds_list.columns[col])):

                file = open(work_path+'%s_%s.txt'%(seeds_list.columns[col], seeds_list.index[row]), 'w')
                inf_spread_matrix = []
                seeds = eval(seeds_list.iloc[row,col])
                for r in range(R):
                    _, scale, _, _, _ = Adaptive_Dissemination.AD(df_hyper_matrix, edge_weights, seeds, t, beta)
                    inf_spread_matrix.append(scale)

                for i in inf_spread_matrix:
                    file.write(' '.join([str(x) for x in i]))
                    file.write('\n')
                file.close()

