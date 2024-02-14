import numpy as np
import networkx as nx
import scipy.linalg as la
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats
from numpy.random.mtrand import dirichlet
import subprocess
import seaborn as sb
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

def cov_to_corr(cov_matrix):

    """Computes correlation matrix from covariance matrix

    :param cov_matrix: covariance matrix
    :type cov_matrix: np.array

    :return: correlation matrix
    :rtype: np.array
    """

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    return Dinv @ cov_matrix @ Dinv

def dir_fun(x,pcount):

    """Diriclet function with parameter scaled by desired pseudocount

    :param x: underlying parameters of distribution (with 0s)
    :type x: np.array

    :param pcount: pseudocount to add to distribution to avoid 0s. Default 1.
    :type pcount: int

    :return: samples
    :rtype: np.array
    """

    a = x+pcount
    f = dirichlet(a)
    return f

def geo_mean(vec):

    """Computes the geometric mean of a vector

    :param vec: vector of numbers
    :type vec: np.array

    :return: geometric mean of the vector
    :rtype: float
    """

    return np.exp(np.log(vec).mean())

def geo_mean_nz(vec):

    """Computes the geometric mean of a vector, ignoring 0 entries

    :param vec: vector of numbers
    :type vec: np.array

    :return: geometric mean of the vector with 0 entries removed
    :rtype: float
    """

    return np.exp(np.log(vec[vec != 0]).mean())

def robust_clr(data):

    """Computes the Robust Centered Log Ratio, which leaves 0s in the data and does not use them in computing CLR

    :param data: Data matrix
    :type data: np.array

    :return: Robust CLR of the data
    :rtype: np.array
    """


    rgeos = np.apply_along_axis(geo_mean_nz,0,data)
    dpls = data.copy()
    whr0 = np.where(data == 0)
    dpls[whr0] = 1
    rclr = np.log(dpls/rgeos)
    rclr[whr0] = 0
    return rclr

def safelog(x,sval = 0):

    """Computes log or alternative return value for 0

    :param x: Argument of log
    :type x: float

    :param sval: return value for log(0)
    :type sval: float

    :return: log(x) or sval
    :rtype: float
    """

    if x > 0:
        return np.log(x)
    else:
        return sval
    
def proj(v1,v2):

    """Projection of vector v1 onto the direction of vector v2
    
    :param v1: vector to be projected
    :type v1: np.array

    :param v2: direction of resulting projection
    :type v2: np.array

    :return: Vector in the direction of v2 (component of v1 in direction v2)
    :rtype: np.array
    """

    if sum(abs(v2)):
        return v2*(np.dot(v2,v1)/np.dot(v2,v2))
    else:
        return v1*0

def rand_in_perp(S,mxvars = 1,minvars = 0.5,bias = 0,pos = None,neg = None):

    """Find a random vector perpendicular to a given set of vectors. L2 norm of the vector will be chosen randomly in (minvars,mxvars). Additionally can 
    choose in or near the dual cone to a set of vectors.

    :param S: Vectors (as rows) that new vector should be perpendicular to
    :type S: np.array

    :param mxvars: maximum scale of vector
    :type mxvars: float

    :param minvars: minimum scale of vector
    :type minvars: float

    :param bias: Extent to bias towards a dual cone. 1 chooses from dual cone, <1 chooses ``close to" the dual cone, 0 does not use the dual cone.
    :type bias: float

    :param pos: set of vectors for which result will have positive dot product (or biased towards)
    :type pos: np.array
    
    :param neg: set of vectors for which result will have negative dot product (or biased towards)
    :type neg: np.array

    :return: random vector
    :rtype: np.array
    """

    if (hasattr(pos,"__len__") or hasattr(neg,"__len__")) and (bias !=0):
        if (hasattr(pos,"__len__") and hasattr(neg,"__len__")):
            K = np.concatenate([pos,-neg],axis = 0)
        elif hasattr(pos,"__len__"):
            K = pos
        else:
            K = -neg
        ### K is the cone of vectors we want to biased towards positive dot product with. First we (may) need
        #project on the null space of the S.
        #need an orthonormal basis perpendicular to the rows of S, i.e. ker(S)
        if S.shape[0]:
            NS = la.null_space(S)
        else:
            NS = np.eye(S.shape[1])
        #now project onto there....
        #So we need a basis for S
        SRng = la.orth(S.T)
        SBasis = np.concatenate([SRng,NS],axis=1) 
        ### And finally the projection:
        K_S = la.solve(SBasis,K.T)
        if K_S.shape[1]:
            K_SPerp = K_S[SRng.shape[1]:]
            ####Next we need a vector from K dual.
            ### This is the vectors y such that K^Ty >= 0
            ## so choose a positive vector (or close to it if the bias is not 1)
            x = (np.random.rand(K_S.shape[1]) - 0.5) + bias/2
            u = la.lstsq(K_SPerp.T,x)[0]
            ### I guess add a random vector from the null space of K_SPerp.T
            NSK = la.null_space(K_SPerp.T)
            u = u + np.dot(NSK,np.random.rand(NSK.shape[1])) #I don't think these 2 lines are necessary...
            r = np.dot(NS,u)
            r = (minvars + (mxvars-minvars)*np.random.rand())*r/np.linalg.norm(r)
        else:
            #then pick a random vector from there.
            r = np.dot(NS,np.random.rand(NS.shape[1])-0.5)
            r = (minvars + (mxvars-minvars)*np.random.rand())*r/np.linalg.norm(r)
    else:
        if S.shape[0]:
            #need an orthonormal basis perpendicular to the rows of S, i.e. ker(S)
            NS = la.null_space(S)
            #then pick a random vector from there.
            r = np.dot(NS,np.random.rand(NS.shape[1])-0.5)
            r = (minvars + (mxvars-minvars)*np.random.rand())*r/np.linalg.norm(r)
        else:
            r = np.random.rand(S.shape[1]) - 0.5
            r = (minvars + (mxvars-minvars)*np.random.rand())*r/np.linalg.norm(r)
    return r

def adjust_sparsity(sparse_adjust,shffld,spl_i):
    inter_king = [shffld[spl_i[i]:spl_i[i+1],spl_i[i]:spl_i[i+1]].sum()/((spl_i[i+1] - spl_i[i])*((spl_i[i+1] - spl_i[i])-1)) for i in range(len(spl_i)-1)]
    blk = shffld[spl_i[sparse_adjust["Kingdoms"][0]]:spl_i[sparse_adjust["Kingdoms"][0]+1],spl_i[sparse_adjust["Kingdoms"][1]]:spl_i[sparse_adjust["Kingdoms"][1]+1]]
    blk_size = blk.size
    desired_sparsity = np.mean(inter_king)*sparse_adjust["SparsityRatio"]
    desired_edges = blk_size*desired_sparsity
    current_edges = blk.sum()
    if sparse_adjust["SparsityRatio"] < 1:
        nz = np.where(blk != 0)
        keep = np.random.choice(nz[0].size,size = int(desired_edges),replace = False)
        keep_coords = (nz[0][keep],nz[1][keep])
        newblk = np.zeros_like(blk)
        newblk[keep_coords] = 1.0
        shffld[spl_i[sparse_adjust["Kingdoms"][0]]:spl_i[sparse_adjust["Kingdoms"][0]+1],spl_i[sparse_adjust["Kingdoms"][1]]:spl_i[sparse_adjust["Kingdoms"][1]+1]] = newblk
        shffld[spl_i[sparse_adjust["Kingdoms"][1]]:spl_i[sparse_adjust["Kingdoms"][1]+1],spl_i[sparse_adjust["Kingdoms"][0]]:spl_i[sparse_adjust["Kingdoms"][0]+1]] = newblk.T
    if sparse_adjust["SparsityRatio"] > 1:
        zers = np.where(blk == 0)
        add_in = np.random.choice(zers[0].size,size = int(desired_edges-current_edges),replace = False)
        add_in_coords = (zers[0][add_in],zers[1][add_in])
        newblk = blk.copy()
        newblk[add_in_coords] = 1
        shffld[spl_i[sparse_adjust["Kingdoms"][0]]:spl_i[sparse_adjust["Kingdoms"][0]+1],spl_i[sparse_adjust["Kingdoms"][1]]:spl_i[sparse_adjust["Kingdoms"][1]+1]] = newblk
        shffld[spl_i[sparse_adjust["Kingdoms"][1]]:spl_i[sparse_adjust["Kingdoms"][1]+1],spl_i[sparse_adjust["Kingdoms"][0]]:spl_i[sparse_adjust["Kingdoms"][0]+1]] = newblk.T
    return shffld

def create_groundtruth_covariance(N,graphtype,sparsity,spl_i,mxvars = 0.5,minvars = 0.4,sparse_adjust = None,bias_str = 0, bias_blocks = None):


    """Creates a randomly chosen covariance matrix (symmetric, positive definite, full rank) with the same non-zero entry structure as the adjacency matrix of a random graph. Uses networkX graph generators. 

    :param N: size of the matrix (result is NxN)
    :type N: int

    :param graphtype: Type of random graph to base the non-zero structure of the matrix on. Choices are ``WS`` for Watts-Strogatz, ``BA`` for Barabosi-Albert, ``ER`` for Erdos-Reyni, and ``PL`` for powerlaw cluster.
    :type graphtype: str
    :param sparsity: approximate proportion of the matrix to be 0
    :type sparsity: float
    :param 
    :param mxvars: maximum range of variance (diagonal entries). Variances are chosen uniformly at random in the interval defined by minvars,mxvars.
    :type mxvars: float
    :param minvars: minimum range of variance (diagonal entries)
    :type minvars: float
    :param sparse_adjust: How to adjust sparsity of different blocks
    :type sparse_adjust: dict
    :param bias_blocks: Details of how to bias the interactions by blocks. List of [(i,j,+/-1)] with i<j
    :type bias_blocks: list
    :param bias_str: strength of biases. 1 is always true, 0 is no bias.
    :type bias_str: float

    :return: _description_
    :rtype: _type_
    """

    nedges = sparsity*(N**2)

    if graphtype == "WS":
        K = int(2*nedges/N)
        p = 0.3
        graph = nx.watts_strogatz_graph(N,K,p)
    elif graphtype == "BA":
        K = int(nedges/N)
        graph = nx.barabasi_albert_graph(N, K)
    elif graphtype == "ER":
        p = 2*nedges/(N*(N-1))
        graph = nx.fast_gnp_random_graph(N, p)
    elif graphtype == "PL":
        p = 0.3
        K = max(int(nedges/(N*(1+p))),1)
        graph = nx.powerlaw_cluster_graph(N, K, p)

    Adj = nx.to_numpy_array(graph)
    sh = np.random.permutation(N)
    shffld = Adj[sh][:,sh]

    if isinstance(sparse_adjust,dict):
        shffld = adjust_sparsity(sparse_adjust,shffld,spl_i)

    Sp = shffld.sum()/(N*(N-1))

    Q = (minvars + (mxvars-minvars))*np.random.rand()*np.eye(N)

    print("[create_groundtruth_covariance] Generating cholesky decomposition")

    if not hasattr(bias_blocks,"__len__"):
        bias_blocks = []

    for j in range(0,len(spl_i)-1):

        pos_blks = [Q[spl_i[k]:spl_i[k+1]] for k in range(j) if (k,j,1) in bias_blocks]
        if len(pos_blks):
            full_pos = np.concatenate(pos_blks,axis = 0)
            pos_pattern = np.concatenate([shffld[:,spl_i[k]:spl_i[k+1]] for k in range(j) if (k,j,1) in bias_blocks],axis = 1)
        else:
            full_pos = np.array([])
        
        neg_blks = [Q[spl_i[k]:spl_i[k+1]] for k in range(j) if (k,j,-1) in bias_blocks]
        if len(neg_blks):
            full_neg = np.concatenate(neg_blks,axis = 0)
            neg_pattern = np.concatenate([shffld[:,spl_i[k]:spl_i[k+1]] for k in range(j) if (k,j,-1) in bias_blocks],axis = 1)
        else:
            full_neg = np.array([])
    

        for i in tqdm(range(spl_i[j],spl_i[j+1])):
            if (j,j,1) in bias_blocks:
                if len(pos_blks):
                    full_pos = np.concatenate([full_pos,Q[spl_i[j]:i]])
                    pos_pattern = np.concatenate([pos_pattern,shffld[:,spl_i[j]:i]],axis = 1)
                else:
                    full_pos = Q[spl_i[j]:i]
                    pos_pattern = shffld[:,spl_i[j]:i]
            elif (j,j,-1) in bias_blocks:
                if len(neg_blks):
                    full_neg = np.concatenate([full_neg,Q[spl_i[j]:i]])
                    neg_pattern = np.concatenate([neg_pattern,shffld[:,spl_i[j]:i]],axis = 1)
                else:
                    full_neg = Q[spl_i[j]:i]
                    neg_pattern = shffld[:,spl_i[j]:i]
            if full_pos.shape[0]:
                pcone = full_pos[pos_pattern[i].astype(bool)]
            else:
                pcone = None
            if full_neg.shape[0]:
                ncone = full_neg[neg_pattern[i].astype(bool)]
            else:
                ncone = None
            Q[i] = rand_in_perp(Q[:i][~shffld[i,:i].astype(bool)],mxvars = mxvars,minvars = minvars,bias = bias_str, pos = pcone,neg = ncone)

    covar = np.dot(Q,Q.T)

    return covar,Sp

def generate_synthetic_samples(NumSmp,covar,mean_log_range = 1,mean_log_center = 0):

    """Generates random "true" synthetic absolute samples by drawing from log-normal distribution

    :param NumSmp: Number of samples to generate
    :type NumSmp: int
    :param covar: Covariance matrix used to define log-normal distribution
    :type covar: np.array
    :param mean_log_range: width of interval to choose mean values for normal distribution (chosen uniformly in given interval) which is afterwards log-transformed
    :type mean_log_range: float
    :mean_log_center: center of interval to choose mean values for normal distribution (chosen uniformly in given interval) which is afterwards log-transformed
    :type mean_log_center: float


    :return: Set of absolute samples
    :rtype: pd.DataFrame
    """


    N = covar.shape[0]
    mean_abundance = mean_log_center + (mean_log_range*np.random.rand(N)- mean_log_range/2)
    try:
        rv = stats.multivariate_normal(mean = mean_abundance,cov = covar)
    except Exception as e:
        print(e.__str__())
        return None,False

    synthetic_truth = pd.DataFrame(index = range(N),columns = ["S{}".format(i) for i in range(NumSmp)])
    print("[generate_synthetic_samples] Drawing {} Samples for {} taxa".format(NumSmp,N))
    for i in tqdm(range(NumSmp)):
        synthetic_truth["S{}".format(i)] = np.exp(rv.rvs())

    return synthetic_truth,True

def simulate_reads(sample,depth):

    """Simulate generating amplicon sequencing reads (error-free) from a absolute sample.

    :param sample: Known absolute abundances of synthetic sample
    :type sample: np.array

    :param depth: Number of reads to generate (sequencing depth)
    :type depth: int

    :return: array of counts indicating number of simulated reads for each synthetic taxa
    :rtype: np.array
    """

    relative_sample = sample/np.sum(sample)
    # reads = np.random.choice(np.arange(len(sample)),p=relative_sample,size = depth)
    # readcounts = np.array([sum(reads == i) for i in np.arange(len(sample))])
    
    readcounts = np.zeros(len(sample))
    for rd in range(depth):
        incr = np.random.choice(np.arange(len(sample)),p = relative_sample)
        readcounts[incr] += 1
    return readcounts

def simulate_all_reads(samples,depths,nj = 1):

    """Simulate generating amplicon sequencing reads (error-free) from a set of absolute samples.

    :param samples: Known absolute abundances of synthetic samples
    :type sample: pd.DataFrame

    :param depths: Number of reads to generate (sequencing depth) for each sample
    :type depth: np.array[int]

    :param nj: number of parallel jobs to run for joblib.Parallel
    :type nj: int

    :return: table of counts indicating number of simulated reads for each synthetic taxa for each sample
    :rtype: pd.DataFrame
    """


    print("[simulate_all_reads] Drawing Read Assignments for {} taxa, {} samples.".format(samples.shape[0],samples.shape[1]))

    all_reads = Parallel(n_jobs=nj)(delayed(simulate_reads)(samples[samples.columns[i]].values,depths[i]) for i in range(len(samples.columns)))

    simulated_counts = pd.DataFrame(np.array(all_reads).T,columns = samples.columns, index=samples.index)

    return simulated_counts.astype(int)

def generate_synthetic_data(num_taxa,num_samples,**kwargs):

    """Main function to generate a sythetic dataset from a log-normal distribution with error-free simulated sequencing.
    
    :param num_taxa: Number of synthetic taxa in the dataset
    :type num_taxa: int
    :param num_samples: Number of synthetic samples to generate
    :type num_samples: int

    :param sparsity: approximate proportion of the covariance matrix to be 0
    :param graph_model: Type of random graph to base the non-zero structure of the matrix on. Choices are ``WS`` for Watts-Strogatz, ``BA`` for Barabosi-Albert, ``ER`` for Erdos-Reyni, and ``PL`` for powerlaw cluster.
    :param max_variance: maximum range of variance (diagonal entries). Variances are chosen uniformly at random in the interval defined by min_variance,max_variance.
    :param min_variance: minimum range of variance (diagonal entries). Variances are chosen uniformly at random in the interval defined by min_variance,max_variance.
    :param mean_log_range: width of interval to choose mean values for normal distribution (chosen uniformly in given interval) which is afterwards log-transformed
    :param mean_log_center: center of interval to choose mean values for normal distribution (chosen uniformly in given interval) which is afterwards log-transformed
    :param data_types: Number of seperate "data types" to simulate - reads will be generated seperately for each
    :param average_read_depth: Average sequencing depth
    :param std_of_read_depth: Std. of sequencing depth
    :param simulate_reads_tasks: number of parallel jobs for simulating reads
    :param truth_sparsity_adjustment: how to adjust sparsity in blocks (in kingdoms or across)
    :param truth_bias_strength: strength of sign bias
    :param truth_bias_structure: structure of sign bias in blocks

    :type sparsity: float
    :type graph_model: str
    :type max_variance: float
    :type min_variance: float
    :type mean_log_range: float
    :type mean_log_center: float
    :type data_types: int
    :type average_read_depth: int
    :type std_of_read_depth: int
    :type simulate_reads_tasks: int


    :return: Synthetic data set including the following fields: "GroundTruthCovariance","GraphSparsity","AbsoluteSamples","CombinedReads","SeparatedReads","GroupOrder","ReadDepths","TotalRatioVariance","DataSplit"
    :rtype: Dict
    """

    sparsity = kwargs.get("sparsity",0.05)
    graphtype = kwargs.get("graph_model","PL")
    mxvar = kwargs.get("max_variance",0.5)
    minvar = kwargs.get("min_variance",0.4)
    mean_lr = kwargs.get("mean_log_range",1)
    mean_lc = kwargs.get("mean_log_center",0)
    sparse_adjust = kwargs.get("truth_sparsity_adjustment",None)
    b_str = kwargs.get("truth_bias_strength",0)
    b_blocks = kwargs.get("truth_bias_structure",None)
    
    num_types = kwargs.get("data_types",1)
    minsplit = int(0.2*num_taxa)
    maxsplit = int(0.8*num_taxa)
    splits = np.sort(np.random.choice(range(minsplit,maxsplit),size = num_types-1,replace = False))

    spl_i = np.concatenate([[0],splits,[num_taxa]])

    nmtry = 0
    giveup = 10
    covar_matrix = np.zeros((num_taxa,num_taxa))
    covar_matrix_done = False
    while not covar_matrix_done:
        covar_matrix, actual_sparsity = create_groundtruth_covariance(num_taxa,graphtype,sparsity,spl_i,mxvars=mxvar,minvars=minvar,bias_blocks=b_blocks,bias_str=b_str,sparse_adjust = sparse_adjust)



        absolute_samples,covar_matrix_done = generate_synthetic_samples(num_samples,covar_matrix,mean_log_range=mean_lr,mean_log_center=mean_lc)

        nmtry += 1
        if nmtry == giveup:
            print("[generate_synthetic_data] Can't draw synthetic after {} tries. Possible ill-conditioned matrix. Matrix smallest singular value is {}.".format(giveup,np.min(np.linalg.svd(covar_matrix)[1])))
            return None

    if not isinstance(absolute_samples,pd.DataFrame):
        return None



    
    avg_dpth = kwargs.get("average_read_depth",20000)
    std_dpth = kwargs.get("std_of_read_depth",1000)

    simr_nj = kwargs.get("simulate_reads_tasks",1)

    read_depths = {}
    
    sep_reads = {}
    reads_order = []
    combined_reads = pd.DataFrame()
    for j in range(1,num_types+1):
        dpths = np.random.normal(avg_dpth,std_dpth,size = num_samples).astype(int)
        read_depths["Set{}".format(j)] = dpths
        rd = simulate_all_reads(absolute_samples.iloc[spl_i[j-1]:spl_i[j]],dpths,nj = simr_nj)
        sep_reads["Set{}".format(j)] = rd
        reads_order += [(spl_i[j-1],spl_i[j])]
        combined_reads = pd.concat([combined_reads,rd])
    
    if num_types == 2:
        spl_at = spl_i[1]
        q = (absolute_samples.iloc[:spl_at]/(sep_reads["Set1"]/sep_reads["Set1"].sum())).replace([np.inf, -np.inf], np.nan).mean()
        r = (absolute_samples.iloc[spl_at:]/(sep_reads["Set2"]/sep_reads["Set2"].sum())).replace([np.inf, -np.inf], np.nan).mean()
        vqr = np.var(q/r)
    else:
        vqr = 0

    return {"GroundTruthCovariance":covar_matrix,"GraphSparsity":actual_sparsity,"AbsoluteSamples":absolute_samples,"CombinedReads":combined_reads,"SeparatedReads":sep_reads,"GroupOrder":reads_order,"ReadDepths":read_depths,"TotalRatioVariance":vqr,"DataSplit":spl_i}

def normalize_table(otu_table,how = "dirichlet",pcount = 1,totcount = 10**5):

    """
    Add pseuodocounts to data so that we can use CLR

    :param otu_table: the count or relative abundance table (taxa x sample)
    :type otu_table: np.array

    :param how: method for adding pseudocounts. Options are ``add_pseudocounts``,``dirichlet`` and ``normalize``
    :type how: str

    :param pcount: pseudocount to add or use for dirichlet distribution. Default 1.
    :type pcount: int
 
    :param totcount: approximate total count to assume if given relative abundance data (data is expanded to behave like count data). Default 10**5
    :type totcount: float
    
    :return: normalized data with no 0s
    :rtype: np.array

    method options are 

    - ``add_pseudocounts`` which adds a small number to each entry (default 1 if data is counts, default 10**4 if data is relative)
    - ``dirichlet`` which, for a vector of counts C, draw the fractions from Dirichlet(C+pcount) or for relative abundance Dirichlet(c*totcount+pcount)
    - ``normalize`` which just divides each column by its sum
    
    """

    ### find out if the data looks like counts or relative abundance

    if np.all(otu_table == otu_table.astype(int)):
        tablekind = 'count'
    else:
        tablekind = 'relative'

    if how.lower() == "add_pseudocounts":
        if tablekind == "count":
            otu_table = otu_table + pcount
            return otu_table/otu_table.sum(axis = 0)
        else:
            otu_table = totcount*otu_table + pcount
            return otu_table/otu_table.sum(axis = 0)
        
    elif how.lower() == "dirichlet":
        if tablekind == "count":
            ### Method is the same as used in SparCC (friedman et al. 2012)
            otu_table = np.apply_along_axis(lambda x:dir_fun(x,pcount),0,otu_table)
            return otu_table
        else:
            otu_table = np.apply_along_axis(lambda x:dir_fun(x,pcount),0,totcount*otu_table)
            return otu_table

    elif how.lower() == "normalize":
        return otu_table/otu_table.sum(axis = 0)

def log_cov(data,normalize = "robust",pcount = 1,totcount = 10**5):

    """Compute the covariance of the log of data

    :param data: Table of data vectors (as columns)
    :type data: np.array
    :param normalize: how the data should be normalized (how 0 entries should be dealt with). Choices are ``robust``, ``add_pseudocounts``,``dirichlet`` and ``normalize``
    :type normalize: = str 
    :param pcount: pseudocount to add or use for dirichlet distribution. Default 1.
    :type pcount: int
 
    :param totcount: approximate total count to assume if given relative abundance data (data is expanded to behave like count data). Default 10**5
    :type totcount: float
    
    :return: covariance matrix of log-transformed data 
    :rtype: np.array

    method options are 

    - ``robust`` which ignores 0s
    - ``add_pseudocounts`` which adds a small number to each entry (default 1 if data is counts, default 10**4 if data is relative)
    - ``dirichlet`` which, for a vector of counts C, draw the fractions from Dirichlet(C+pcount) or for relative abundance Dirichlet(c*totcount+pcount)
    - ``normalize`` which just divides each column by its sum

    """

    if normalize == "robust":
        log_data = np.vectorize(safelog)(data)#data.applymap(safelog)
    else:
        normed_data = normalize_table(data,how = normalize,pcount = pcount,totcount = totcount)
        log_data = np.vectorize(safelog)(normed_data)#.applymap(safelog)

    niave_cov = np.cov(log_data)
    
    return niave_cov

def compute_clr(otu_table, norm = "dirichlet",pcount = 1,totcount = 10**5):

    """
    :param otu_table: the count or relative abundance table (taxa x sample)
    :type otu_table: np.array

    :param norm: method for adding pseudocounts. Options are ``add_pseudocounts``,``dirichlet`` and ``normalize``. See :py:func:`normalize_table <compute_clr.normalize_table>`
    :type norm: str

    :param pcount: See :py:func:`normalize_table <compute_clr.normalize_table>`
    :type pcount: int
 
    :param totcount: See :py:func:`normalize_table <compute_clr.normalize_table>`
    :type totcount: float
    
    :return: CLR of the data (by column)
    :rtype: np.array
    """
    if norm == "robust":
        print("[compute_clr] computing robust clr transform")
        clr_table = robust_clr(otu_table)
    else:
        print("[compute_clr] normalizing data using {} method".format(norm))
        normalized_data = normalize_table(otu_table, how=norm, pcount = pcount, totcount = totcount)

        print("[compute_clr] computing clr transform")
        clr_table = np.log(normalized_data/(np.apply_along_axis(geo_mean,0,normalized_data)))

    return clr_table

def clr_cov(otu_table,**kwargs):

    """
    Make a covariance matrix from an otu_table. Can handle data from mixed datatypes with the split_indices parameter, which indicates which rows in the table belong to each data type. The result is the matrix :math:`\\Gamma` from 
    the Kurtz et al.

    :param otu_table: OTU table given as counts or relative abundance
    :type otu_table: numpy array

    :param split_indices: tuple with indicies corresponding to each data type, so that CLR can be computed separately (see Tipton et al. 2018)
    :type split_indices: tuple

    :param kwargs: kwargs for :py:func:`compute_clr<make_cov_table.compute_clr>`

    :return: CLR covariance matrix :math:`\\Gamma`
    :rtype: np.array

    """

    otu_table = otu_table.values

    norm = kwargs.get("norm","dirichlet")
    pcount = kwargs.get("pcount",1)
    totcount = kwargs.get("totcount",10**5)
    split_indices = kwargs.get("split_indices",())

    if not len(split_indices):
        split_indices = (range(otu_table.shape[0]),)

    clrd_data = np.zeros(otu_table.shape)

    for smptype in split_indices:
        clrd_data[smptype] = compute_clr(otu_table[smptype],norm=norm,pcount=pcount,totcount=totcount)#[smptype]


    return np.cov(clrd_data)

def sparcc(otu_table,normtype = "dirichlet"):

    """Uses SparCC3 to infer covariance of data vectors.

    :param otu_table: Data table with columns corresponding to samples.
    :type otu_table: np.array or pd.DataFrame
    :param normtype: method for adding pseudocounts. Options are ``add_pseudocounts``,``dirichlet`` and ``normalize``. See :py:func:`normalize_table <compute_clr.normalize_table>`
    :type normtype: str
    :return: inferred covariance matrix
    :rtype: np.array
    """


    import SparCC3 as spc
    print("[sparcc] computing covariance.")
    if normtype == "robust":
        if not (otu_table == 0).any().any():
            res =  spc.main(otu_table.T,oprint = False,norm= "normalize")
        else:
            print("SparCC does not support robust CLR for tables with 0s. Using pseudocount")
            res =  spc.main(otu_table.T,oprint = False,norm= "pseudo")
    elif normtype  == "add_pseudocounts":
        res =  spc.main(otu_table.T,oprint = False,norm= "pseudo")
    elif normtype == "dirichlet":
        res =  spc.main(otu_table.T,oprint = False,norm= "dirichlet")
    elif normtype == "normalize":
        res =  spc.main(otu_table.T,oprint = False,norm= "normalize")
    return res[1]



