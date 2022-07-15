import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

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

def annualized_return(rets, periods_per_year):
    number_periods = rets.shape[0]
    return np.prod(1+rets)**(periods_per_year/number_periods) - 1

def annualized_volatility(rets):
    number_periods= rets.shape[0]
    return rets.std()*(number_periods**0.5)

def sharpe_ratio(rets, rfr, periods_per_year):
    r = annualized_return(rets, periods_per_year)
    v = annualized_volatility(rets)
    return (r-rfr)/v

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
        
def gaussian_VaR(r, level=5, cornish=False):
    s, k = skewness_kurtosis(r)
    z = norm.ppf(level/100)
    if cornish:
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def portfolio_return(weights, er):
    return np.matmul(weights.T, er)

def portfolio_volatility(weights, cov):
    return (weights.T @ cov@ weights)**0.5

def plot_ef_2(er, cov, n_points):
    if cov.shape[0] > 2:
        raise TypeError('More than two assets')
        
    weights = [np.array((w,1-w)) for w in np.linspace(0,1,n_points)]
    r = [tk.portfolio_return(w, er) for w in weights]
    v = [tk.portfolio_volatility(w, er) for w in weights]
    ef = pd.DataFrame({'Return':r, 'Volatility':v})
    return ef.plot.scatter(x='Volatility', y='Return')

def minimize_vol(target_return, cov, er):
    
    n = cov.shape[0]
    bounds = ((0.0,1.0),)*n
    init_weights = np.repeat(1/n, n)
    r_to_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda er, weights: target_return - portfolio_return(weights, er)
    }
    w_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    weights = minimize(portfolio_volatility, 
                   init_weights, args=(cov,), 
                   method='SLSQP', 
                   options={'disp':False}, 
                   bounds=bounds, 
                   constraints=(r_to_target, w_to_1)
                   )
    
    return weights.x

def optimal_weights(cov, er, n_points):
    t_r = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(r, cov, er) for r in t_r]
    return weights

def plot_ef(cov, er, n_points, cml=False, rfr=0):
    
    weights = optimal_weights(cov, er, n_points)
    r = [portfolio_return(w, er) for w in weights]
    v = [portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({'Return':r, 'Volatility':v})
    ax = ef.plot.line(x='Volatility', y='Return', style='.-')
    
    if cml:
        ax.set_xlim(left=0)
        w_msr = msr(rfr, cov, er)
        vol_msr = portfolio_volatility(w_msr, cov)
        r_msr = portfolio_return(w_msr, er)
        x_vals = [0, vol_msr]
        y_vals =[rfr, r_msr]
        ax.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2, marker='o', markersize=12)
    
    return ax
    
def msr(rfr, cov, er):
    n = cov.shape[0]
    bounds = ((0.0,1.0),)*n
    init_weights = np.repeat(1/n, n)
    w_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def negative_sharpe_ratio(weights, rfr, cov, er):
        r = portfolio_return(weights, er)
        vol = portfolio_volatility(weights, cov)
        return -(r-rfr)/vol
    
    weights = minimize(negative_sharpe_ratio, 
                   init_weights, args=(rfr, cov, er,), 
                   method='SLSQP', 
                   options={'disp':False}, 
                   bounds=bounds, 
                   constraints=(w_to_1)
                   )
    
    return weights.x
    
    