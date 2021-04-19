from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict

def run_PCA(X,y,title):
    
    pca = PCA(random_state=5).fit(X) #for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Explained Variance and Eigenvalues: "+ title)
    fig.tight_layout()
    plt.show()
    
def run_ICA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: "+ title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.show()

def run_RCA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i,dim in product(range(5),dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()


    fig, ax1 = plt.subplots()
    ax1.plot(dims,mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims,std_recon, 'm-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: "+ title)
    fig.tight_layout()
    plt.show()
    
def run_RFC(X,y,df_original):
    rfc = RFC(n_estimators=500,min_samples_leaf=round(len(X)*.01),random_state=5,n_jobs=-1)
    imp = rfc.fit(X,y).feature_importances_ 
    imp = pd.DataFrame(imp,columns=['Feature Importance'],index=df_original.columns[2::])
    imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum']<=0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols

loanX,loanY,telescopeX,telescopeY = import_data()
run_PCA(loanX,loanY,"Loan Data")
run_ICA(loanX,loanY,"Loan Data")
run_RCA(loanX,loanY,"Loan Data")
imp_loan, topcols_loan = run_RFC(loanX,loanY,df_loan)

loanX,loanY,telescopeX,telescopeY = import_data()
run_PCA(telescopeX,telescopeY,"Telescope Data")
run_ICA(telescopeX,telescopeY,"Telescope Data")
run_RCA(telescopeX,telescopeY,"Telescope Data")
imp_telescope, topcols_telescope = run_RFC(telescopeX,telescopeY,df_telescope)

loanX,loanY,telescopeX,telescopeY = import_data()
imp_loan, topcols_loan = run_RFC(loanX,loanY,df_loan)
pca_loan = PCA(n_components=4,random_state=5).fit_transform(loanX)
ica_loan = ICA(n_components=8,random_state=5).fit_transform(loanX)
rca_loan = ICA(n_components=6,random_state=5).fit_transform(loanX)
rfc_loan = df_loan[topcols_loan]
rfc_loan = np.array(rfc_loan.values,dtype='int64')

run_kmeans(pca_loan,loanY,'PCA loan Data')
run_kmeans(ica_loan,loanY,'ICA loan Data')
run_kmeans(rca_loan,loanY,'RCA loan Data')
run_kmeans(rfc_loan,loanY,'RFC loan Data')

evaluate_kmeans(KMeans(n_clusters=10,n_init=10,random_state=100,n_jobs=-1),pca_loan,loanY)
evaluate_kmeans(KMeans(n_clusters=10,n_init=10,random_state=100,n_jobs=-1),ica_loan,loanY)
evaluate_kmeans(KMeans(n_clusters=5,n_init=10,random_state=100,n_jobs=-1),rca_loan,loanY)
evaluate_kmeans(KMeans(n_clusters=10,n_init=10,random_state=100,n_jobs=-1),rfc_loan,loanY)

run_EM(pca_loan,loanY,'PCA Loan Data')
run_EM(ica_loan,loanY,'ICA Loan Data')
run_EM(rca_loan,loanY,'RCA Loan Data')
run_EM(rfc_loan,loanY,'RFC Loan Data')

evaluate_EM(EM(n_components=20,covariance_type='diag',n_init=1,warm_start=True,random_state=100),pca_loan,loanY)
evaluate_EM(EM(n_components=18,covariance_type='diag',n_init=1,warm_start=True,random_state=100),ica_loan,loanY)
evaluate_EM(EM(n_components=20,covariance_type='diag',n_init=1,warm_start=True,random_state=100),rca_loan,loanY)
evaluate_EM(EM(n_components=22,covariance_type='diag',n_init=1,warm_start=True,random_state=100),rfc_loan,loanY)

loanX,loanY,telescopeX,telescopeY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(telescopeX),np.array(telescopeY), test_size=0.2)
run_PCA(X_train,telescopeY,"Telescope Data")
run_ICA(X_train,telescopeY,"Telescope Data")
run_RCA(X_train,telescopeY,"Telescope Data")
imp_telescope, topcols_telescope = run_RFC(X_train,y_train,df_telescope)

pca_telescope = PCA(n_components=6,random_state=5).fit_transform(X_train)
ica_telescope = ICA(n_components=8,random_state=5).fit_transform(X_train)
rca_telescope = ICA(n_components=7,random_state=5).fit_transform(X_train)
rfc_telescope = df_telescope[topcols_telescope]
rfc_telescope = np.array(rfc_telescope.values,dtype='int64')

evaluate_kmeans(KMeans(n_clusters=11,n_init=10,random_state=100,n_jobs=-1),pca_telescope,y_train)
evaluate_kmeans(KMeans(n_clusters=8,n_init=10,random_state=100,n_jobs=-1),ica_telescope,y_train)
evaluate_kmeans(KMeans(n_clusters=9,n_init=10,random_state=100,n_jobs=-1),rca_telescope,y_train)
evaluate_kmeans(KMeans(n_clusters=10,n_init=10,random_state=100,n_jobs=-1),rfc_telescope,telescopeY)

evaluate_EM(EM(n_components=8,covariance_type='diag',n_init=1,warm_start=True,random_state=100),pca_telescope,y_train)
evaluate_EM(EM(n_components=18,covariance_type='diag',n_init=1,warm_start=True,random_state=100),ica_telescope,y_train)
evaluate_EM(EM(n_components=15,covariance_type='diag',n_init=1,warm_start=True,random_state=100),rca_telescope,y_train)
evaluate_EM(EM(n_components=16,covariance_type='diag',n_init=1,warm_start=True,random_state=100),rfc_telescope,telescopeY)

