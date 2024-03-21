import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import scipy.stats as stats
from sklearn import metrics
import networkx as nx
from cdlib.algorithms import louvain
from tqdm import tqdm
from scipy.integrate import solve_ivp
from numba import njit
from joblib import Parallel,delayed

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
        if split_locations.loc[i,0]:
            all_covs[i] = load_trial_covariance(i)
    return all_covs,split_locations

def load_all_correlation():
    split_locations = pd.read_csv(os.path.join("synthetic_data","split_locations.csv"),index_col = 0)
    split_locations.columns = split_locations.columns.astype(int)
    L = len(split_locations)
    all_covs = {}
    for i in range(L):
        if split_locations.loc[i,0]:
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
    return rmse_df.astype(float)

def make_XK_RMSE_df(trial_set,splits,true_column = "Ground Truth"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rmse_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        rmse = calculate_RMSE(edge_df)
        rmse_df.loc[ky] = rmse.loc[true_column,fit_methods_li]
    return rmse_df.astype(float)

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
            rmse.loc[r,c] = (stats.linregress(dfcpy[r], dfcpy[c]).rvalue**2)
    return rmse

def make_coeff_det_df(trial_set,true_column = "Ground Truth",cutoff = 0,cutoff_method="zero"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rsqrd_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_edge_df(tr)
        rsqrd = calculate_coeff_det(edge_df,cutoff=cutoff,cutoff_method=cutoff_method)
        rsqrd_df.loc[ky] = rsqrd.loc[true_column,fit_methods_li]
    return rsqrd_df.astype(float)

def make_XK_coeff_det_df(trial_set,splits,true_column = "Ground Truth",cutoff = 0,cutoff_method = "zero"):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    rsqrd_df = pd.DataFrame(index = trial_set.keys(),columns = fit_methods_li)
    for ky,tr in trial_set.items():
        edge_df = make_XK_edge_df(tr,splits.loc[ky,0])
        rsqrd = calculate_coeff_det(edge_df,cutoff = cutoff,cutoff_method=cutoff_method)
        rsqrd_df.loc[ky] = rsqrd.loc[true_column,fit_methods_li]
    return rsqrd_df.astype(float)

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

def make_centrality_df(net_set,posnet = False):
    centrality_df = pd.DataFrame(columns = net_set.keys())
    for ky,val in net_set.items():
        if posnet:
            gr = np.zeros_like(val)
            gr[val > 0] = val[val > 0]
        else:
            gr = val
        centrality_df[ky] = eig_centrality(np.abs(gr))
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

def all_hub_performance_topN(trial_set,N=-1,hubct=0.5,posnet = False):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    auc_df = pd.DataFrame(columns = fit_methods_li)
    roc_crvs = dict([(m,[]) for m in fit_methods_li])
    for ky,tr in trial_set.items():
        cent_df = make_centrality_df(tr,posnet = posnet)
        roc_sc = hub_performance(cent_df,N=N,hubct=hubct)
        auc_df.loc[ky] = roc_sc[0]
        for m in fit_methods_li:
            roc_crvs[m] += [roc_sc[1][m][:2]]
    return auc_df.astype(float),roc_crvs

def all_hub_performance_byval(trial_set,cuts = np.linspace(0,1,10),posnet = False):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    perf_df = pd.DataFrame(columns = np.concatenate([[m]*len(trial_set) for m in fit_methods_li]),index = cuts)
    for c in cuts:
        tmpdf = all_hub_performance_topN(trial_set,N=-1,hubct = c,posnet = posnet)[0]
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

def get_positive_nx_graph(gr,cutoff = 0):
    gr_plus = np.zeros_like(gr)
    gr_plus[gr > cutoff] = gr[gr > cutoff]
    g = nx.from_numpy_array(gr_plus)
    return g

def get_unsigned_nx_graph(gr, cutoff = 0):
    gr_cut = np.zeros_like(gr)
    gr_cut[abs(gr) > cutoff] = gr[abs(gr) > cutoff]
    g = nx.from_numpy_array(abs(gr_cut))
    return g

def get_centralities(nxgr,degnxgr = None):

    if not isinstance(degnxgr,nx.Graph):
        degnxgr = nxgr
    
    N = len(nxgr.nodes)
    
    bcent = nx.betweenness_centrality(nxgr,weight = 'weight')
    bcent_v = [bcent[i] for i in range(N)]
    
    dcent = nx.degree_centrality(degnxgr)
    dcent_v = [dcent[i] for i in range(N)]
    
    try:
        ecent = nx.eigenvector_centrality(nxgr,weight = 'weight')
        ecent_v = [ecent[i] for i in range(N)]
    except:
        ecent_v = np.zeros(N)
        print("[get_centralities] WARNING: Could not compute eigenvector (probable power iteraction failure)")
    
    return bcent_v,dcent_v,ecent_v

def compare_centrality(trial,posnet = True,degree_cutoff_q = 0.9):
    graphs = ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split']
    if posnet:
        grnd_truth = get_positive_nx_graph(trial["Ground Truth"])
    else:
        grnd_truth = get_unsigned_nx_graph(trial["Ground Truth"])
    grnd_truth_cent = get_centralities(grnd_truth)
    centrality_rsq_df = pd.DataFrame(index = graphs, columns = ["Betweenness","Degree","Eigenvector"])
    for gr in graphs:
        degcut = np.quantile(abs(trial[gr]).flatten(),degree_cutoff_q)
        if posnet:
            ngr = get_positive_nx_graph(trial[gr])
            deg_ngr = get_positive_nx_graph(trial[gr],cutoff=degcut)
        else:
            ngr = get_unsigned_nx_graph(trial[gr])
            deg_ngr = get_unsigned_nx_graph(trial[gr],cutoff=degcut)
        gr_cent = get_centralities(ngr,degnxgr = deg_ngr)
        centrality_rsq_df.loc[gr] = [stats.linregress(grnd_truth_cent[0],gr_cent[0]).rvalue**2,stats.linregress(grnd_truth_cent[1],gr_cent[1]).rvalue**2,stats.linregress(grnd_truth_cent[2],gr_cent[2]).rvalue**2]
    return centrality_rsq_df

def centrality_rocs(trial,trueq = 0.95, trueN = -1,posnet = True,degree_cutoff_q = 0.9):
    graphs = ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split']
    if posnet:
        grnd_truth = get_positive_nx_graph(trial["Ground Truth"])
    else:
        grnd_truth = get_unsigned_nx_graph(trial["Ground Truth"])    
    grnd_truth_cent = get_centralities(grnd_truth)
    if trueN == -1:
        true_hubs = {"Betweenness":(grnd_truth_cent[0] >= np.quantile(grnd_truth_cent[0],trueq)).astype(int),"Degree":(grnd_truth_cent[1] >= np.quantile(grnd_truth_cent[1],trueq)).astype(int),"Eigenvector":(grnd_truth_cent[2] >= np.quantile(grnd_truth_cent[2],trueq)).astype(int)}
    else:
        true_hubs = {"Betweenness":(grnd_truth_cent[0] >= np.sort(grnd_truth_cent[0])[-trueN]).astype(int),"Degree":(grnd_truth_cent[1] >= np.sort(grnd_truth_cent[1])[-trueN]).astype(int),"Eigenvector":(grnd_truth_cent[2] >= np.sort(grnd_truth_cent[2])[-trueN]).astype(int)}
    centrality_roc_df = pd.DataFrame(index = graphs, columns = ["Betweenness","Degree","Eigenvector"])
    roc_curves = {}
    for gr in graphs:
        degcut = np.quantile(abs(trial[gr]).flatten(),degree_cutoff_q)
        if posnet:
            ngr = get_positive_nx_graph(trial[gr])
            deg_ngr = get_positive_nx_graph(trial[gr],cutoff=degcut)
        else:
            ngr = get_unsigned_nx_graph(trial[gr])
            deg_ngr = get_unsigned_nx_graph(trial[gr],cutoff=degcut)   
        gr_cent = get_centralities(ngr,degnxgr = deg_ngr)
        roc_curves[gr] = {"Betweenness":metrics.roc_curve(true_hubs["Betweenness"],gr_cent[0]),"Degree":metrics.roc_curve(true_hubs["Degree"],gr_cent[1]),"Eigenvector":metrics.roc_curve(true_hubs["Eigenvector"],gr_cent[2])}
        try:
            bet = metrics.roc_auc_score(true_hubs["Betweenness"],gr_cent[0])
        except ValueError:
            bet = 0.5
        try:
            deg = metrics.roc_auc_score(true_hubs["Degree"],gr_cent[1])
        except ValueError:
            deg = 0.5
        try:
            eig = metrics.roc_auc_score(true_hubs["Eigenvector"],gr_cent[2])
        except ValueError:
            eig = 0.5
        centrality_roc_df.loc[gr] = [bet,deg,eig]
    return centrality_roc_df,roc_curves


def centrality_ROC_alltrials(trialset,posnet = True,degree_cutoff_q = 0.9,trueq = 0.95, trueN = -1):
    betweenness_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    betweenness_curves = dict([(m,[]) for m in betweenness_all_trials.index])
    degree_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    degree_curves = dict([(m,[]) for m in degree_all_trials.index])
    eigen_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    eigen_curves = dict([(m,[]) for m in eigen_all_trials.index])
    for ky,val in trialset.items():
        trialdf,trialcurves = centrality_rocs(val,posnet = posnet,degree_cutoff_q = degree_cutoff_q,trueq = trueq, trueN = trueN)
        
        betweenness_all_trials[ky] = trialdf["Betweenness"]
        for m in betweenness_all_trials.index:
            betweenness_curves[m] += [trialcurves[m]["Betweenness"][:2]]
            
        degree_all_trials[ky] = trialdf["Degree"]
        for m in degree_all_trials.index:
            degree_curves[m] += [trialcurves[m]["Degree"][:2]]
            
        eigen_all_trials[ky] = trialdf["Eigenvector"]
        for m in eigen_all_trials.index:
            eigen_curves[m] += [trialcurves[m]["Eigenvector"][:2]]
    return betweenness_all_trials,degree_all_trials,eigen_all_trials,{"Betweenness":betweenness_curves,"Degree":degree_curves,"Eigenvector":eigen_curves}





def compare_centrality_alltrials(trialset,posnet = True,degree_cutoff_q = 0.9):
    betweenness_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    degree_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    eigen_all_trials = pd.DataFrame(index =  ['Log-Covariance', 'CLR-Mixed', 'CLR-Split', 'SparCC', 'GLASSO-Mixed', 'GLASSO-Split'], columns = trialset.keys())
    for ky,val in trialset.items():
        trialdf = compare_centrality(val,posnet = posnet,degree_cutoff_q = degree_cutoff_q)
        betweenness_all_trials[ky] = trialdf["Betweenness"]
        degree_all_trials[ky] = trialdf["Degree"]
        eigen_all_trials[ky] = trialdf["Eigenvector"]
    return betweenness_all_trials,degree_all_trials,eigen_all_trials

def cluster_overlap(net1,net2):
    cls_1 = louvain(net1,randomize = False)
    cls_2 = louvain(net2,randomize = False)
    clusters_1_overlap_DF = pd.DataFrame(columns = ["Largest Overlap","Most Contained In","Most Containing"])
    for i in range(len(cls_1.communities)):
        c = cls_1.communities[i]
        clusters_1_overlap_DF.loc[i] = max([len(set(c).intersection(set(cl)))/len(set(c).union(set(cl))) for cl in cls_2.communities]),max([len(set(c).intersection(set(cl)))/len(cl) for cl in cls_2.communities]),max([len(set(c).intersection(set(cl)))/len(c) for cl in cls_2.communities])
    clusters_2_overlap_DF = pd.DataFrame(columns = ["Largest Overlap","Most Contained In","Most Containing"])
    for i in range(len(cls_2.communities)):
        c = cls_2.communities[i]
        clusters_2_overlap_DF.loc[i] = max([len(set(c).intersection(set(cl)))/len(set(c).union(set(cl))) for cl in cls_1.communities]),max([len(set(c).intersection(set(cl)))/len(cl) for cl in cls_1.communities]),max([len(set(c).intersection(set(cl)))/len(c) for cl in cls_1.communities])
    return clusters_1_overlap_DF,clusters_2_overlap_DF

def overlap_all_nets(netset):
    clusters_overlap_means = pd.DataFrame(columns = ["Largest Overlap","Most Contained In","Most Containing"])
    grnd_truth = get_positive_nx_graph(netset["Ground Truth"])
    for ky,val in netset.items():
        nx_Graph = get_positive_nx_graph(val)
        overlap,_ = cluster_overlap(nx_Graph,grnd_truth)
        clusters_overlap_means.loc[ky] = overlap.mean()
    return clusters_overlap_means

@njit
def lv_sys(t,s,adj):
    return s*(1+np.dot(adj,s))

def compute_key_sample(taxa,sample,network,endtime = 20):##called 500*8*number_trials times
    sol1 = solve_ivp(lv_sys,[0, endtime],sample,args = (network,))#simulate_LV(sample.values,network,self_inhibit = self_inhibit)
    weight = sample[taxa]
    sample[taxa] = 0
    sol2 = solve_ivp(lv_sys,[0, endtime],sample,args = (network,))#simulate_LV(sample_cp.values,network,self_inhibit = self_inhibit)
    sol2_ep = sol2.y[:,-1]
    sol2_ep[sol2_ep == 0] = 1 #if the solution is identically 0, then it must have started there and so will be 0 in sol1 as well, or be the taxa we set to 0. Either way we want a log-ratio of 0.
    log_ratios = np.log(np.divide(sol2_ep,sol1.y[:,-1],where = sol1.y[:,-1] !=0,out = np.ones_like(sol1.y[:,-1])))
    return log_ratios,weight

def compute_key_network_jl(taxa,all_samples,network,nj = 1,number_trials = -1):#called 500*8 times
    if number_trials < 0:
        number_trials = all_samples.shape[1]
    all_logratio = np.empty((all_samples.shape[0],number_trials))#
    weights = np.empty(number_trials)#
    sampleres = np.array(Parallel(n_jobs = nj)(delayed(compute_key_sample)(taxa,all_samples[:,s],network,endtime = 20) for s in range(number_trials)))
    for i in range(number_trials):
        all_logratio[:,i] = sampleres[i][0]
        weights[i] = sampleres[i][1]
    return all_logratio,weights

def compute_keystoneness(taxa,all_samples,network,nj = 1,number_trials = -1):### Called 500*8 times
    lrs,we = compute_key_network_jl(taxa,all_samples,network,nj = nj,number_trials = number_trials)
    return np.dot((lrs.sum(axis = 0)/(lrs.shape[0] - 1)),we)

def compute_all_keystoneness_network(all_samples,network,nj = 1,number_trials = -1):##Called 8 times
    all_kyness = np.empty(len(all_samples))
    for i in tqdm(range(all_samples.shape[0])):###for every taxa (all samples is taxa x sample) 500*8 calls
        all_samples_cp = all_samples.copy()
        all_samples_cp[i,:] = all_samples_cp[i,:] + np.mean(all_samples_cp) #put the taxa of interest in every sample
        all_kyness[i] = compute_keystoneness(i,all_samples_cp,network,nj = nj,number_trials = number_trials)
    return all_kyness

def compute_all_keystoneness(all_samples,network_set,self_inhibit = False,nj = 1,number_trials = -1):
    all_kyness = pd.DataFrame(index = all_samples.index,columns = network_set.keys())
    for net in all_kyness.columns:
        print(net)
        adj = network_set[net].copy()
        adj = adj/(np.abs(adj).max())
        if self_inhibit:
            np.fill_diagonal(adj,-np.abs(adj).sum(axis = 1))
        all_kyness[net] = compute_all_keystoneness_network(all_samples.values,adj,nj = nj,number_trials = number_trials)
    return all_kyness

def keystones(all_trials,random_ic = True,random_sparsity = 0.3,nj = 1, number_trials = 10,self_inhibit = False):
    all_keystones = {}
    for i,trial in all_trials.items():
        if random_ic:
            initial_conditions_arr = np.random.rand(trial['Ground Truth'].shape[0],number_trials)
            zero_mask = np.random.choice([0,1],p = [random_sparsity,1-random_sparsity],size = initial_conditions_arr.shape)
            initial_conditions = pd.DataFrame(initial_conditions_arr*zero_mask)
        else:
            samples1 = pd.read_csv(os.path.join('synthetic_data','simulated_read_data','trial{}'.format(i),'Set1.csv'),index_col = 0)
            samples2 = pd.read_csv(os.path.join('synthetic_data','simulated_read_data','trial{}'.format(i),'Set2.csv'),index_col = 0)
            initial_conditions = pd.concat([samples1/samples1.sum(),samples2/samples2.sum()])[np.random.choice(samples1.columns,min(number_trials,len(samples1.columns)),replace = False)]
        all_keystones[i] = compute_all_keystoneness(initial_conditions,trial,nj = nj,number_trials = number_trials,self_inhibit = self_inhibit)
    return all_keystones

def keystone_performance(keystone_df,N = -1,hubq = 0.95,side = 'positive'):
    kdf = keystone_df.copy()
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    if side == 'negative':
        kdf = -kdf
    elif side == 'absolute':
        kdf = kdf.abs()
        
    if N == -1:
        ct = np.quantile(kdf["Ground Truth"],hubq)
    else:
        ct = np.sort(kdf["Ground Truth"])[-N]

    kdf["Ground Truth Binary"] = (kdf["Ground Truth"] > ct).astype(int)

    auc_array = np.empty(len(fit_methods_li))
    roc_curves = {}
    for i,meth in enumerate(fit_methods_li):
        auc_array[i] = metrics.roc_auc_score(kdf["Ground Truth Binary"],kdf[meth])
        roc_curves[meth] = metrics.roc_curve(kdf["Ground Truth Binary"],kdf[meth])
    return auc_array,roc_curves

def all_keystone_performance(keystones_set,N=-1,hubq = 0.95,side = 'positive'):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    auc_df = pd.DataFrame(columns = fit_methods_li)
    roc_crvs = dict([(m,[]) for m in fit_methods_li])
    for ky,tr in keystones_set.items():
        roc_sc = keystone_performance(tr,N=N,hubq=hubq,side = side)
        auc_df.loc[ky] = roc_sc[0]
        for m in fit_methods_li:
            roc_crvs[m] += [roc_sc[1][m][:2]]
    return auc_df.astype(float),roc_crvs

def keystone_performance_linearfit(keystone_scores_df):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    return [stats.linregress(keystone_scores_df[c], keystone_scores_df["Ground Truth"]).rvalue**2 for c in fit_methods_li]

def all_keystone_performance_LF(keystones_set):
    fit_methods_li = ["Log-Covariance","CLR-Mixed","CLR-Split","SparCC","GLASSO-Mixed","GLASSO-Split"]
    linear_fit_r_df = pd.DataFrame(columns = fit_methods_li)
    for ky,tr in keystones_set.items():
        linear_fit_r_df.loc[ky] = keystone_performance_linearfit(tr)
    return linear_fit_r_df.astype(float)

def bias(all_edges_in,split):
    all_edges_arr = all_edges_in.copy()
    if isinstance(all_edges_arr,pd.DataFrame):
        all_edges_arr = all_edges_arr.values
    grp1_edges = all_edges_arr[:split,:split]
    grp1_edges = grp1_edges[np.triu_indices_from(grp1_edges,k=1)]
    grp2_edges = all_edges_arr[split:,split:]
    grp2_edges = grp2_edges[np.triu_indices_from(grp2_edges,k=1)]
    xk_edges = all_edges_arr[split:,:split].flatten() #upper corner
    tt_dict = {"G1G2":stats.mannwhitneyu(grp1_edges/np.max(np.abs(grp1_edges)),grp2_edges/np.max(np.abs(grp2_edges))).pvalue,"G1XK":stats.mannwhitneyu(grp1_edges/np.max(np.abs(grp1_edges)),xk_edges/np.max(np.abs(xk_edges))).pvalue,"G2XK":stats.mannwhitneyu(xk_edges/np.max(np.abs(xk_edges)),grp2_edges/np.max(np.abs(grp2_edges))).pvalue}
    mean_dict = {"G1_Mean":np.mean(grp1_edges/np.max(np.abs(grp1_edges))),"G2_Mean":np.mean(grp2_edges/np.max(np.abs(grp2_edges))),"XK_Mean":np.mean(xk_edges/np.max(np.abs(xk_edges)))}
    return mean_dict,tt_dict