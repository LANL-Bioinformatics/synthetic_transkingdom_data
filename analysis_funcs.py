import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import scipy.stats as stats
from sklearn import metrics

def cov_to_corr(cov_matrix):
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    return Dinv @ cov_matrix @ Dinv

def load_trial_covariance(tr):
    trial_cov = {}
    fit_methods = {"Log-Covariance":"log_covariance","CLR-Mixed":"CLR_mixed","CLR-Split":"CLR_split","SparCC":"sparcc","GLASSO-Mixed":"glasso_mixed","GLASSO-Split":"glasso_split"}
    trial_cov["Ground Truth"] = pd.read_csv(os.path.join("synthetic_data","true_covariance","trial{}.csv".format(tr)),index_col = 0).values
    abs_abund = pd.read_csv(os.path.join("synthetic_data","true_abundance","trial{}.csv".format(tr)),index_col=0).values
    trial_cov["Absolute Emperical"] = np.cov(abs_abund)
    for ky,val in fit_methods.items():
        trial_cov[ky] = pd.read_csv(os.path.join(val,"trial{}".format(tr),"covariance.csv"),index_col = 0).values
    return trial_cov

def load_trial_correlation(tr):
    trial_cor = {}
    fit_methods = {"Log-Covariance":"log_covariance","CLR-Mixed":"CLR_mixed","CLR-Split":"CLR_split","SparCC":"sparcc","GLASSO-Mixed":"glasso_mixed","GLASSO-Split":"glasso_split"}
    trial_cor["Ground Truth"] = pd.read_csv(os.path.join("synthetic_data","true_correlation","trial{}.csv".format(tr)),index_col = 0).values
    abs_abund = pd.read_csv(os.path.join("synthetic_data","true_abundance","trial{}.csv".format(tr)),index_col=0).values
    trial_cor["Absolute Emperical"] = cov_to_corr(np.cov(abs_abund))
    for ky,val in fit_methods.items():
        trial_cor[ky] = pd.read_csv(os.path.join(val,"trial{}".format(tr),"correlation.csv"),index_col = 0).values
    return trial_cor

def load_all_covariance():
    split_locations = pd.read_csv(os.path.join("synthetic_data","split_locations.csv"),index_col = 0)
    split_locations.columns = split_locations.columns.astype(int)
    L = len(split_locations)
    all_covs = {}
    for i in range(L):
        all_covs[i] = load_trial_covariance(i)
    return all_covs,split_locations

def load_all_correlation():
    split_locations = pd.read_csv(os.path.join("synthetic_data","split_locations.csv"),index_col = 0)
    split_locations.columns = split_locations.columns.astype(int)
    L = len(split_locations)
    all_covs = {}
    for i in range(L):
        all_covs[i] = load_trial_correlation(i)
    return all_covs,split_locations

def make_edge_df(net_set):
    edge_df = pd.DataFrame(columns = net_set.keys())
    for ky,val in net_set.items():
        edge_df[ky] = val[np.triu_indices_from(val,k=1)]
    return edge_df

def make_XK_edge_df(net_set,split):
    edge_df = pd.DataFrame(columns = net_set.keys())
    for ky,val in net_set.items():
        edge_df[ky] = val[split:,:split].flatten()
    return edge_df

def calculate_RMSE(df):
    dfcols = pd.DataFrame(columns=df.columns)
    rmse = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            rmse.loc[r,c] = np.mean((df[r] - df[c]).values**2)**(1/2)
    return rmse

def make_RMSE_df(trial_set,true_column = "Ground Truth"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rmse_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_edge_df(tr)
        rmse = calculate_RMSE(edge_df)
        rmse_df.loc[ky] = rmse.loc[true_column,fit_methods_li]
    return rmse_df

def make_XK_RMSE_df(trial_set,splits,true_column = "Ground Truth"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rmse_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        rmse = calculate_RMSE(edge_df)
        rmse_df.loc[ky] = rmse.loc[true_column,fit_methods_li]
    return rmse_df

def calculate_coeff_det(df,cutoff = 0,cutoff_method = "zero"):
    dfcols = pd.DataFrame(columns=df.columns)
    rmse = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            dfcpy = df.copy()
            if cutoff_method.lower() == "zero":
                dfcpy.loc[dfcpy[c].abs() < cutoff*dfcpy[c].abs().max(),c] = 0
            elif cutoff_method.lower() == "ignore":
                dfcpy = dfcpy.loc[dfcpy[c].abs() >= cutoff*dfcpy[c].abs().max()]
            rmse.loc[r,c] = (stats.linregress(dfcpy[r], dfcpy[c]).rvalue)
    return rmse

def make_coeff_det_df(trial_set,true_column = "Ground Truth",cutoff = 0,cutoff_method="zero"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rsqrd_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_edge_df(tr)
        rsqrd = calculate_coeff_det(edge_df,cutoff=cutoff,cutoff_method=cutoff_method)
        rsqrd_df.loc[ky] = rsqrd.loc[true_column,fit_methods_li]
    return rsqrd_df

def make_XK_coeff_det_df(trial_set,splits,true_column = "Ground Truth",cutoff = 0,cutoff_method = "zero"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rsqrd_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        rsqrd = calculate_coeff_det(edge_df,cutoff = cutoff,cutoff_method=cutoff_method)
        rsqrd_df.loc[ky] = rsqrd.loc[true_column,fit_methods_li]
    return rsqrd_df

def topN_accuracy(edge_df,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    accuracy = np.empty(len(fit_methods_li))
    for i,m in enumerate(fit_methods_li):
        mdf = edge_df[["Ground Truth",m]]
        topNcut = np.sort(mdf[m].abs().values)[-N]
        topNdf = mdf[mdf[m].abs() >= topNcut]
        signdf = np.sign(topNdf.round(8))
        accuracy[i] = (signdf["Ground Truth"] == signdf[m]).sum()/len(signdf)
    return accuracy

def top_accuracy(edge_df,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    accuracy_df = pd.DataFrame(index = range(1,N+1),columns = fit_methods_li)
    for n in range(1,N+1):
        accuracy_df.loc[n] = topN_accuracy(edge_df,n)
    return accuracy_df

def all_top_accuracy(trial_set,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    accuracy_df = pd.DataFrame()
    for tr in trial_set.values():
        edge_df = make_edge_df(tr)
        tmpdf = top_accuracy(edge_df,N)
        accuracy_df = pd.concat([accuracy_df,tmpdf],axis = 1)
    return accuracy_df.astype(float)

def all_top_XK_accuracy(trial_set,splits,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    accuracy_df = pd.DataFrame()
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        tmpdf = top_accuracy(edge_df,N)
        accuracy_df = pd.concat([accuracy_df,tmpdf],axis = 1)
    return accuracy_df.astype(float)

def unsigned_classifier_performance(edge_df,trueN):
    if trueN == -1:
        ct = 10**-8
    else:
        ct = np.sort(edge_df["Ground Truth"].abs())[-trueN]
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    edf = edge_df.copy()
    edf["Ground Truth Binary"] = (edf["Ground Truth"].abs() > ct).astype(int)
    auc_array = np.empty(len(fit_methods_li))
    roc_curves = {}
    for i,meth in enumerate(fit_methods_li):
        auc_array[i] = metrics.roc_auc_score(edf["Ground Truth Binary"],edf[meth].abs())
        roc_curves[meth] = metrics.roc_curve(edf["Ground Truth Binary"],edf[meth].abs())
    return auc_array,roc_curves
        
def all_classifier_performance_topN(trial_set,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    auc_df = pd.DataFrame(columns = fit_methods_li)
    roc_crvs = dict([(m,[]) for m in fit_methods_li])
    for ky,tr in trial_set.items():
        edge_df = make_edge_df(tr)
        roc_sc = unsigned_classifier_performance(edge_df,N)
        auc_df.loc[ky] = roc_sc[0]
        for m in fit_methods_li:
            roc_crvs[m] += [roc_sc[1][m][:2]]
    return auc_df.astype(float),roc_crvs

def all_XK_classifier_performance_topN(trial_set,splits,N):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    auc_df = pd.DataFrame(columns = fit_methods_li)
    roc_crvs = dict([(m,[]) for m in fit_methods_li])
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        roc_sc = unsigned_classifier_performance(edge_df,N)
        auc_df.loc[ky] = roc_sc[0]
        for m in fit_methods_li:
            roc_crvs[m] += [roc_sc[1][m][:2]]
    return auc_df.astype(float),roc_crvs

def all_classifier_performance(trial_set,Ns = [50,100,200,500,-1]):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    perf_df = pd.DataFrame(columns = np.concatenate([[m]*len(trial_set) for m in fit_methods_li]),index = Ns)
    for n in Ns:
        tmpdf = all_classifier_performance_topN(trial_set,n)[0]
        tmpdf = tmpdf.unstack().to_frame().T
        tmpdf.columns = tmpdf.columns.map(lambda l:l[0])
        perf_df.loc[n] = tmpdf.loc[0]
    return perf_df.astype(float)

def all_XK_classifier_performance(trial_set,splits,Ns = [50,100,200,500,-1]):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    perf_df = pd.DataFrame(columns = np.concatenate([[m]*len(trial_set) for m in fit_methods_li]),index = Ns)
    for n in Ns:
        tmpdf = all_XK_classifier_performance_topN(trial_set,splits,n)[0]
        tmpdf = tmpdf.unstack().to_frame().T
        tmpdf.columns = tmpdf.columns.map(lambda l:l[0])
        perf_df.loc[n] = tmpdf.loc[0]
    return perf_df.astype(float)


def eig_centrality(net):
    evals,evecs = np.linalg.eigh(net)
    cent = evecs[:,np.argmax(evals)]
    return cent/cent[np.argmax(np.abs(cent))]

def make_centrality_df(net_set):
    centrality_df = pd.DataFrame(columns = net_set.keys())
    for ky,val in net_set.items():
        centrality_df[ky] = eig_centrality(np.abs(val))
    return centrality_df

def hub_performance(centrality_df,N = -1,hubct = 0.5):
    if N == -1:
        ct = hubct
    else:
        ct = np.sort(centrality_df["Ground Truth"].abs())[-N]
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    cdf = centrality_df.copy()
    cdf["Ground Truth Binary"] = (cdf["Ground Truth"] > ct).astype(int)
    auc_array = np.empty(len(fit_methods_li))
    roc_curves = {}
    for i,meth in enumerate(fit_methods_li):
        auc_array[i] = metrics.roc_auc_score(cdf["Ground Truth Binary"],cdf[meth])
        roc_curves[meth] = metrics.roc_curve(cdf["Ground Truth Binary"],cdf[meth])
    return auc_array,roc_curves

def all_hub_performance_topN(trial_set,N=-1,hubct=0.5):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    auc_df = pd.DataFrame(columns = fit_methods_li)
    roc_crvs = dict([(m,[]) for m in fit_methods_li])
    for ky,tr in trial_set.items():
        cent_df = make_centrality_df(tr)
        roc_sc = hub_performance(cent_df,N=N,hubct=hubct)
        auc_df.loc[ky] = roc_sc[0]
        for m in fit_methods_li:
            roc_crvs[m] += [roc_sc[1][m][:2]]
    return auc_df.astype(float),roc_crvs

def all_hub_performance_byval(trial_set,cuts = np.linspace(0,1,10)):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    perf_df = pd.DataFrame(columns = np.concatenate([[m]*len(trial_set) for m in fit_methods_li]),index = cuts)
    for c in cuts:
        tmpdf = all_hub_performance_topN(trial_set,N=-1,hubct = c)[0]
        tmpdf = tmpdf.unstack().to_frame().T
        tmpdf.columns = tmpdf.columns.map(lambda l:l[0])
        perf_df.loc[c] = tmpdf.loc[0]
    return perf_df.astype('float')

def all_hub_performance_byN(trial_set,Ns = [50,100,200,500]):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    perf_df = pd.DataFrame(columns = np.concatenate([[m]*len(trial_set) for m in fit_methods_li]),index = Ns)
    for n in Ns:
        tmpdf = all_hub_performance_topN(trial_set,N=n,hubct = 0.5)[0]
        tmpdf = tmpdf.unstack().to_frame().T
        tmpdf.columns = tmpdf.columns.map(lambda l:l[0])
        perf_df.loc[n] = tmpdf.loc[0]
    return perf_df.astype('float')

def unweight(net,thr):
    return np.sign(net)*(abs(net) >= thr)
def topQnet(net_o,q=0.9):
    net = net_o.copy()
    if isinstance(net,pd.DataFrame):
        net = net.values
    qstr = np.quantile(np.abs(np.tril(net,k=-1).flatten()),q)
    net = net - np.diag(np.diag(net))
    return unweight(net,qstr)

def make_mean_edge_str_df(trial_set):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    mean_es_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_edge_df(tr)
        mean_es_df.loc[ky] = edge_df.abs().mean()
    return mean_es_df

def make_XK_mean_edge_str_df(trial_set,splits):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    mean_es_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        mean_es_df.loc[ky] = edge_df.abs().mean()
    return mean_es_df