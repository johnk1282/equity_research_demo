import multiprocessing as mp
import pandas as pd

def multi_groupby(df,groupby_cols,func):
    #helper function to do a pandas groupby().apply() using multi-processing
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(func,[df_sub for _,df_sub in df.groupby(groupby_cols)])
    pool.close()
    pool.join()
    return pd.concat(results)

