import pandas as pd
import numpy as np

class dataloader:
    node_dict = {}

    def __init__(self, matrix_path, weights_path):
        self.matrix_path = matrix_path
        self.weights_path = weights_path

    def dataload(self):
        df = pd.read_csv(self.matrix_path, index_col=False, header=None)
        arr = df.values
        node_list = []
        for each in arr:
            node_list.extend(list(map(int, each[0].split(" "))))
        node_arr = np.unique(np.array(node_list))

        for i in range(0, len(node_arr)):
            self.node_dict[node_arr[i]] = i
        self.node_num = len(list(node_arr))
        self.hp_edge_num = len(arr)

        matrix = np.random.randint(0, 1, size=(self.node_num, self.hp_edge_num))
        index = 0
        for each in arr:
            edge_list = list(map(int, each[0].split(" ")))
            for edge in edge_list:
                matrix[self.node_dict[edge]][index] = 1
            index = index + 1
        self.hyper_matrix = pd.DataFrame(matrix)

        float_list = []
        with open(self.weights_path, 'r') as file:
            for line in file:
                float_list.append(float(line.strip()))
        self.edge_weights = np.array(float_list)
        
        
