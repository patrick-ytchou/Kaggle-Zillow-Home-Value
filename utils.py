# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# Timer decorator
def timeit(method):
    """
    Helper function for timing.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('----------------------')
            print('Function %r takes %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


# Resume table for quick dataframe examination 
def resumetable(df, features):
    """
    Helper function to create descriptive statistics summary.
    
    Parameters
    ----------
    df : dataset you want to take a look at.
    """
    
    # Extract the dataset you are interested in.
    df = df[features]
    
    # Start creating descriptive statistics summary.
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Missing %'] = 100 * df.isnull().sum().values / len(df)
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return(summary)


# Functions for memory reduction
def reduce_mem_usage(df, verbose=True):
    """
    Helper function to transform data type into smaller ones, to make sure local machine doesn't crash.
    
    Parameters
    -----------
    df : dataset you want to transform
    verbose : whether or not you want to print out memory reduction.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return(df)


# For categorical features
def compare_value_counts(feature):
    """
    Helper function to compare value counts of two datasets.
    
    Parameters
    ----------
    feature : feature that you want to deep dive into.
    """
    df1 = pd.DataFrame(df_prop_2016[feature].value_counts()).T
    df2 = pd.DataFrame(df_prop_2017[feature].value_counts()).T
    dd = pd.concat([df1, df2])
    print(f'{len(dd.columns)} unique values for feature "{feature}": {list(dd.columns)}')
    return(dd)

# For numerical features
def plot_distribution(feature):
    """
    Helper function to plot distribution.
    
    Parameters
    -----------
    feature : feature that you want to deep dive into.
    
    """
    plt.figure(figsize=(8,6))
    sns.distplot(df_prop_2016[feature], hist=False)
    sns.distplot(df_prop_2017[feature], hist=False)
    

def plot_scatter(df, x, y, facet=None):
    """
    Helper function for scatter plot.
    
    Parameters
    ----------
    df : dataframe you are interested in.
    x : x-axis in the scatter plot.
    y : y-axis in the scatter plot.
    facet : facet to wrap on.
    """
    if facet:
        return(ggplot(aes(x = x, y = y, color=facet), data=df) + geom_jitter() + \
               facet_wrap(facets=facet) + geom_smooth(method = 'lm', color='red'))
    else:
        return(ggplot(aes(x = x, y = y), data=df) + geom_jitter() + geom_smooth(method = 'lm', color='red'))
    

# Load dataset
def load_data(filepath):
    """
    Helper function to load dataset.
    
    Parameters
    ----------
    filepath : filepath of the csv file.
    """
    df = pd.read_csv(filepath)
    df = reduce_mem_usage(df)
    print(f"Number of rows: {len(df)}".format(len(df)))
    print(f"Number of columns: {len(df.columns)-1} \n")
    return df
