import math
import numpy as np
import os
import pandas as pd



def read_csv_to_df(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print("An error occurred while reading the CSV file:")
        print(str(e))


def df_to_data(data_frame):
    #required DataFrame column/order: datetime/exposue/rate/sell_rate/buy_rate
    df = data_frame
    exposure = df.iloc[:,1]
    rate = df.iloc[:,2]
    sell_rate = df.iloc[:,3]
    buy_rate = df.iloc[:,4]
    return exposure, rate, sell_rate, buy_rate