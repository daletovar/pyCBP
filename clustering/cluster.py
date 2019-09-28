import numpy as np
import nibabel as nib  
import scipy.sparse as ss
import os
import dask
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .utils import html_table




class Silly:

    def __init__(self, directory, shape=(91,109,91)):

        # check for existing files

        row,col,data = np.loadtxt(file, unpack=True)
        self.matrix =  ss.csr_matrix((data.astype(np.float64, (
            row.astype(np.int32,col.astype(np.int32))))))
        self.shape = shape
        self.coords = coords
        self.reference_space = reference_space

    def _repr_html_(self):
        html_table(self)

    def fit_predict(self, model=KMeans, n_clusters=4):
        
        self.model = model(n_clusters=n_clusters)
        #dense = dask.delayed()
        dense = self.matrix.toarray()
        cc = np.corrcoef(dense)
        cc = np.nan_to_num(cc)

        all_labels = []
        silhouettes = []
        for n in n_clusters:
            all_labels.append(model(n_clusters=n).fit_predict(cc) + 1)
            silhouettes.append(silhouette_score(all_labels[-1],cc))
        self.silhouettes = silhouettes
        brains = [np.zeros(self.shape)]
        for labels, brain in zip(all_labels, brains):
            brain[coords] = labels
        final = np.stack(brains, axis=3)
        header = self.reference_space.header
        header['cal_min'] = final.min()
        header['cal_max'] = final.max()
        self.nifti = nib.nifti1.Nifti1Image(final,
                affine=self.reference_space.affine, header=header)
        
        return self.nifti


