import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm

def preprocess_returns(file_path):
    """preprocessing csv file:
        1. get date as index
        2. eliminate days in date
        3. turn percentages into decimals   
    """
    rets = pd.read_csv(file_path, index_col=0)
    rets.index = pd.to_datetime(rets.index).to_period('M')
    return rets/100

def monthly_return(df):
    return np.prod(1+ df)**(1/df.shape[0]) -1

def drawdown(df, init_wealth):
    wealth_index = init_wealth*(1+df).cumprod()
    previous_peaks = wealth_index.cummax()
    return pd.DataFrame({'wealth': np.array(wealth_index), 
                         'peaks': np.array(previous_peaks), 
                         'drawdown':-(wealth_index-previous_peaks)/previous_peaks})

def semideviation(rets):
    negative = rets[rets<0].std()
    positive = rets[rets>=0].std()
    return negative, positve

def skewness_kurtosis(rets):
    mean_zero = rets - rets.mean()
    sigma = rets.std(ddof=0)
    skewness = ((mean_zero**3).mean())/sigma**3
    kurtosis = ((mean_zero**4).mean())/sigma**4
    return skewness, kurtosis

def is_normal(ret, alpha=0.05):
    statistic, p_value = scipy.stats.jarque_bera(ret)
    return p_value > alpha

def historic_VaR(r,level):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(historic_VaR, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('not a dataframe or series')
       
def historic_cVaR(r,level):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(historic_cVaR, level=level)
    elif isinstance(r, pd.Series):
        beyond_vals = r <= -historic_VaR(r, level)
        return -r[beyond_vals].mean()
    else:
        raise TypeError('not a dataframe or series')
        
def gaussian_VaR(r, level, cornish=False):
    s, k = skewness_kurtosis(r)
    z = norm.ppf(level/100)
    if cornish:
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))