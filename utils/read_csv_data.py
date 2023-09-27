import pandas as pd
import numpy as np

def read_csv_data(data_path: str):
    pandas_data = pd.read_csv(data_path)
    data = np.array(pandas_data.values)
    name = list(pandas_data.columns)
    name_dict = {}
    for i in range(len(name)):
        name_dict[i] = name[i]
    
    return name_dict, data
    

if __name__ == "__main__":
    name_dict, data = read_csv_data("/home/bennie/bennie/temp/machine2deeplearning_lab/dataset/npvproject-concrete.csv")
    print(name_dict, data.shape)
    