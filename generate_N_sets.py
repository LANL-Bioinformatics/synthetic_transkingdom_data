import numpy as np
import networkx as nx
import scipy.linalg as la
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats
import SparCC3 as spc
from numpy.random.mtrand import dirichlet
import subprocess
import matplotlib.pyplot as plt
import seaborn as sb
from joblib import Parallel, delayed
from pathlib import Path
import os
import datetime
import pickle as pk


from synthetic_fit_funcs import *

if __name__=="__main__":


    dt_stmp = datetime.datetime.now().strftime("m%m_d%d_t%H.%M")
    rfld = os.path.join("results",dt_stmp)
    Path(rfld).mkdir(parents=True, exist_ok=True)

    N = 500
    M = 150

    lam = 0.3
    ch = 0
    ns = 1 # 2 for data split in half. Can also split more times...
    gsp = 0.005

    mxvar = 2
    mnvar =0.3

    norms = "add_pseudocounts"
    grmod = "PL"

    mean_spread = 4 #for choosing the mean of the log-data higher for larger spread (e.g. more rare & dominant taxa)
    mean_center = 0 #to center things around 1...

    L = 5

    simulate_reads_tasks = 6

    hubcut = 0.3

    avgdpth = 100000

    stddpth = 10000
    use_correlation = False

    sparse_adjust = {"Kingdoms":(0,1),"SparsityRatio":0.1}
    bias_blocks = [(0,1,1)] #cross kingdom interactions more positive
    bias_str = 0.2 #but not a lot more.


    with open(os.path.join(rfld,"parameters.txt"),'w') as fl:
        fl.write("N (number of taxa): {}\n".format(N))
        fl.write("M (number of samples): {}\n".format(M))
        fl.write("lam: {} (LASSO parameter lambda for GLASSO)\n".format(lam))
        fl.write("ch: {} (chunkiness for NormalCowboy)\n".format(ch))
        fl.write("ns: {} (Number of simualted data types)\n".format(ns))
        fl.write("gsp: {} (approximate sparsity of ground truth)\n".format(gsp))
        fl.write("norms: {} (how count data is normalized by the method)\n".format(norms))
        fl.write("grmod: {} (which graph model to base ground truth on)\n".format(grmod))
        fl.write("mean_spread: {} (spread of log-means of absolute abundances)\n".format(mean_spread))
        fl.write("mean_center: {} (center of uniform distrubition of log-means of absolute abundances)\n".format(mean_center))
        fl.write("L (number of trials): {}\n".format(L))
        fl.write("mxvar: {} (Max variance of absuolute abundances)\n".format(mxvar))
        fl.write("mnvar: {} (Min variance of absuolute abundances)\n".format(mnvar))
        fl.write("hubcut (eigenvector centrality cutoff to consider a node a hub for comparison between networks): {}\n".format(hubcut))
        fl.write("average read depth: {}\nStandard deviation of read depth: {}\n".format(avgdpth,stddpth))
        fl.write("Use Correlation (instead of covariance): {}\n".format(use_correlation))
        fl.write("Sparsity Structure Adjustment: {}\n".format(sparse_adjust))
        fl.write("Block sign bias structure: {}\n".format(bias_blocks))
        fl.write("Sign Bias Strength: {}\n".format(bias_str))



    Path(os.path.join(rfld,"synthetic_data","true_covariance")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rfld,"synthetic_data","true_correlation")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rfld,"synthetic_data","true_abundance")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rfld,"synthetic_data","simulated_read_data")).mkdir(parents=True, exist_ok=True)



    sparsity_df = pd.Series(index = range(L),dtype = 'float64')

    read_depth_set1_df = pd.DataFrame(index = range(L), columns = ["S{}".format(s) for s in range(M)])

    for tr in range(L):
        kwargs = {
            'max_variance':max_var,
            'min_variance':min_var,
            'data_types':ns,
            'graph_model':grmod,
            'sparsity':gsp,
            'mean_log_range':mean_spread,
            'mean_log_center':mean_center,
            'simulate_reads_tasks':simulate_reads_tasks,
            'average_read_depth':avgdpth, 
            'std_of_read_depth':stddpth,
            'truth_sparsity_adjustment':sparse_adjust,
            'truth_bias_strength':bias_str,
            'truth_bias_structure':bias_blocks
        }
        synthetic_data = generate_synthetic_data(N,M,**kwargs)

        pd.DataFrame(synthetic_data["GroundTruthCovariance"]).to_csv(os.path.join(rfld,"synthetic_data","true_covariance","trial{}.csv".format(tr)))
        pd.DataFrame(cov_to_corr(synthetic_data["GroundTruthCovariance"])).to_csv(os.path.join(rfld,"synthetic_data","true_correlation","trial{}.csv".format(tr)))

        synthetic_data["AbsoluteSamples"].to_csv(os.path.join(rfld,"synthetic_data","true_abundance","trial{}.csv".format(tr)))

        # Path(os.path.join(rfld,"synthetic_data","simulated_read_data","trial{}".format(tr))).mkdir(parents=True, exist_ok=True)
        # for ky,rds in synthetic_data["SeparatedReads"].items():
            # rds.to_csv(os.path.join(rfld,"synthetic_data","simulated_read_data","trial{}".format(tr),"{}.csv".format(ky)))
        synthetic_data["CombinedReads"].to_csv(os.path.join(rfld,"synthetic_data","simulated_read_data","trial{}.csv".format(tr)))


        sparsity_df.loc[tr] = synthetic_data["GraphSparsity"]
        read_depth_set1_df.loc[tr] = synthetic_data["ReadDepths"]["Set1"]


    sparsity_df.to_csv(os.path.join(rfld,"synthetic_data","graph_sparsity.csv"))

    Path(os.path.join(rfld,"synthetic_data","simulated_read_data","read_depths")).mkdir(parents=True, exist_ok=True)

    read_depth_set1_df.to_csv(os.path.join(rfld,"synthetic_data","simulated_read_data","read_depths","set1.csv"))


