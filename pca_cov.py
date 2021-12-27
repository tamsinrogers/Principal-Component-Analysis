'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Tamsin Rogers
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''perform and store principal component analysis results'''
class PCA_COV:

    def __init__(self, data):

        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        return self.prop_var

    def get_cum_var(self):
        return self.cum_var

    def get_eigenvalues(self):
        return self.e_vals

    def get_eigenvectors(self):
        return self.e_vecs

    '''returns the covariance matrix of data'''
    def covariance_matrix(self, data):
        
        Ac = data-data.mean(axis=0)
        n = Ac.shape[0]
        result = (1 / (n-1)) * (Ac.T@Ac)

        return result

    '''returns the proportion variance accounted for by the principal components (PCs)'''
    def compute_prop_var(self, e_vals):

        vars = []
        s = np.sum(e_vals)
        
        for i in e_vals:
            vars.append(i/s)

        self.prop_var = vars
        return vars

    '''returns the cumulative variance accounted for by the principal components (PCs)'''
    def compute_cum_var(self, prop_var):
    
        vars = []
        total = 0
        
        for i in prop_var:
            total += i
            vars.append(total)
            
        self.cum_var = vars
        return vars
        
    '''performs PCA on the given datav ariables vars'''
    def pca(self, vars, normalize=False):
        
        self.vars = vars
        self.A = self.data.loc[:,vars]

        if normalize is True:
            self.copyA = self.A.copy().to_numpy()
            self.A = (self.A - self.A.min()) / (self.A.max() - self.A.min())
            self.normalized = True
        
        covmat = self.covariance_matrix(self.A)
        (evals,evecs) = np.linalg.eig(covmat)
        sort = np.argsort(evals)[::-1]
        self.e_vals = evals
        self.e_vecs = evecs
        
        self.prop_var = self.compute_prop_var(self.get_eigenvalues())
        self.cum_var = self.compute_cum_var(self.get_prop_var())

    '''plots cumulative variance accounted for by the top num_pcs_to_keep PCs'''
    def elbow_plot(self, num_pcs_to_keep=None):
        
        cv = self.get_cum_var()
        
        if num_pcs_to_keep is None:
            points = cv
        else:
            points = cv[:num_pcs_to_keep]
        
        plt.plot(points,'bx-')
        plt.title(f"Cumulative variance accounted for by the top {num_pcs_to_keep} PCs", fontsize=20)
        plt.xlabel("# of PCs")
        plt.ylabel("Cumulative Variance")

    '''projects the data onto pcs_to_keep PCs'''
    def pca_project(self, pcs_to_keep):
        
        evec = self.get_eigenvectors()
        vhat = evec[:,pcs_to_keep]
        Ac = self.A-self.A.mean(axis=0)
        self.A_proj = Ac @vhat
        
        return self.A_proj
        
    '''projects the data into the PCA space on the top_k PCs and then projects it back to the data space'''
    def pca_then_project_back(self, top_k):

        evec = self.get_eigenvectors()
        vhat = evec[:,:top_k]
        num = np.arange(0, top_k)
        proj = self.pca_project(num)
        
        if self.normalized is True:
            
            mean = np.array(self.copyA.mean(axis = 0))
            ran = self.copyA.max(axis = 0) - self.copyA.min(axis = 0)
            result = ran * (proj@vhat.T) + mean
            
        else:
            mean = np.array(self.A.mean(axis = 0))
            result = (proj@vhat.T) + mean
            
        return result
        
