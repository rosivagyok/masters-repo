import pandas as pd
import numpy as np
import numpy.random as rd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_trafo(pose_feats,comp):
    n_comp = comp
    pca_trafo = PCA(n_components=n_comp)
    data = pose_feats

    pca_data = pca_trafo.fit_transform(data)
    pca_inv_data = pca_trafo.inverse_transform(pca_data)

    """plt.figure(figsize = (10,6.5));
    plt.semilogy(pca_trafo.explained_variance_ratio_, '--o');
    plt.xlabel('principal component', fontsize = 20);
    plt.ylabel('explained variance', fontsize = 20);
    plt.tick_params(axis='both', which='major', labelsize=10);
    plt.tick_params(axis='both', which='minor', labelsize=10);
    plt.xlim([0, 66]);"""

    z_scaler = StandardScaler()
    z_data = z_scaler.fit_transform(data)
    pca_trafo2 = PCA().fit(z_data);

    fig, ax1 = plt.subplots(figsize = (10,6.5))
    ax1.set_xlabel('principal component', fontsize = 10);

    
    plt.legend(loc=(0.01, 0.075) ,fontsize = 10);

    ax2 = ax1.twinx()
    ax2.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), '--go', label = 'cumulative explained variance ratio');
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    ax2.tick_params(axis='both', which='minor', labelsize=10);
    ax2.tick_params(axis='both', which='minor', labelsize=10);
    plt.xlim([0, 66]);
    plt.legend(loc=(0.01, 0),fontsize = 18);

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(pca_trafo.inverse_transform(np.eye(n_comp)), cmap="hot", cbar=False)
    plt.ylabel('principal component', fontsize=20);
    plt.xlabel('original feature index', fontsize=20);
    plt.tick_params(axis='both', which='major', labelsize=10);
    plt.tick_params(axis='both', which='minor', labelsize=10);

    fig = plt.figure(figsize=(12, 10))
    plt.plot(pca_inv_data.mean(axis=0), '--o', label = 'mean')
    plt.plot(np.square(pca_inv_data.std(axis=0)), '--o', label = 'variance')
    plt.legend(loc='lower right')
    plt.ylabel('feature contribution', fontsize=20);
    plt.xlabel('feature index', fontsize=20);
    plt.tick_params(axis='both', which='major', labelsize=10);
    plt.tick_params(axis='both', which='minor', labelsize=10);
    plt.xlim([0, 66])
    plt.legend(loc='lower left', fontsize=10)
    plt.show()
    return fig