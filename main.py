
import pandas as pd
import database as db
import datapreparation as dp
import datapreparation2 as dp2
import models
import neural_network as nn
import dask.dataframe as dd


def main():
    data = db.get_all('data_models')
    # print(data.info())
    #data = db.aggregate_fun()
    #print(data)
    
    # ddf.to_dask_array(lengths=True)
    #models.just_dask(df)
    models.dask_model(data)
    # nn.neural_network(data)
main()