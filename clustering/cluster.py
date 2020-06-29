import numpy as np
import nibabel as nib  
import scipy.sparse as ss
import os
import pickle
import dask.array as da
from dask import delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Iterable

def validate_paths(dirs):

    def validate_dir(dir):
        files = ['coords_for_fdt_matrix2', 'fdt_matrix2.dot']
        for f in files:
            full_path = os.path.join(dir,f)
            if not os.path.exists(full_path):
                raise IOError('%s not found.' % full_path)
    
    if isinstance(dirs, str):
        group = False
        nsubs = 1
        if not os.path.isdir(dirs):
            raise IOError('%s is not a directory.' %(dir))
        validate_dir(dirs)
    elif isinstance(dirs, Iterable):
        nsubs = len(dirs)
        group = True
        for d in dirs:
            if not os.path.isdir(d):
                raise IOError('%s is not a directory.' %(str(d)))
            validate_dir(d)
    else:
        raise ValueError('%s must be a path or list of paths.' %(dirs))
    return nsubs, group

def load_group(dirs):
    out = load_sub(dirs[0])
    for d in dirs[1:]:
        out += load_sub(d)
    return out

def load_sub(d):
    print('loading %s' % d)
    row,col,data = np.loadtxt(os.path.join(d, 'fdt_matrix2.dot'), unpack=True)
    data = data.astype(np.float64)
    row = row.astype(np.int32)
    col = col.astype(np.int32)
    out = ss.csr_matrix((data, (
            row,col)))[1:,1:]
    return out

def toarray(x):
    out = np.zeros(x.shape)
    for i, (start,stop) in enumerate(zip(x.indptr[:-1], x.indptr[1:])):
        out[i,x.indices[start:stop]] = x.data[start:stop]
    return out


class CBP:

    def __init__(self, directories, roi_mask, matrix=None):

        if matrix is None:
            nsubs, group = validate_paths(directories)
            if group: self.matrix = load_group(directories)
            else: self.matrix = load_sub(directories)
            
        else:
            nsubs = 1
            if isinstance(matrix, ss.spmatrix):
                self.matrix = matrix
            else:
                try:
                    self.matrix = ss.load_npz(matrix)
                except:
                    ValueError('%s must be an npz file.' % matrix)

        
        if isinstance(roi_mask,str):
            try:
                self.space = nib.load(roi_mask)
                self.shape = self.space.get_data().shape
            except:
                IOError()
        elif isinstance(roi_mask, nib.Nifti1Image):
            self.space = roi_mask
            self.shape = self.space.get_data().shape
        else:
            ValueError()
        if nsubs==1:
            self.coords = np.loadtxt(os.path.join(directories,'coords_for_fdt_matrix2'))[:,:3].astype(np.int32)
        else:
            self.coords = np.loadtxt(os.path.join(directories[0],'coords_for_fdt_matrix2'))[:,:3].astype(np.int32)
        self.nsubs = nsubs
        self.corr_mat = None
        self.fitted = False
        self.silhouettes = []               

    @property
    def silhouette_table(self):
        return SilhouetteTable(self.silhouettes)
    
    def __repr__(self):
        return 'CBP: nsubs={}, brain_space_shape={}, fitted={}'.format(
            self.nsubs,self.shape,self.fitted)

    
    def fit(self):
        if self.corr_mat is None:
            dense = da.from_delayed(delayed(toarray)(self.matrix),
                shape=self.matrix.shape,dtype=self.matrix.dtype)
            self.corr_mat = da.corrcoef(dense).compute()
        self.fitted = True
    
    def predict(self, model=KMeans, n_clusters=4):
        self.model = model(n_clusters=n_clusters)
        labels = self.model.fit_predict(self.corr_mat) + 1
        sil = [type(self.model).__name__, n_clusters, silhouette_score(self.corr_mat, labels)]
        if not sil in self.silhouettes:
            self.silhouettes.append(sil)
        out = np.zeros(self.shape)
        out[self.coords[:,0],self.coords[:,1],self.coords[:,2]] = labels
        header = self.space.header
        header['cal_min'] = out.min()
        header['cal_max'] = out.max()
        self.nifti = nib.nifti1.Nifti1Image(out,
                affine=self.space.affine, header=header)
        return self.nifti

    def fit_predict(self, model=KMeans, n_clusters=4):
        self.fit()
        return self.predict(model=model, n_clusters=n_clusters)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['matrix']
        if self.corr_mat is not None:
            del state['corr_mat']
        return state
    
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.matrix = ss.load_npz(os.path.join(self.path,'mat.npz'))
        if os.path.exists(os.path.join(self.path,'corr_mat.npz')):
            self.corr_mat = np.load(os.path.join(self.path,'corr_mat.npz'))

    def save(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        ss.save_npz(os.path.join(path, 'mat.npz'),self.matrix)
        if self.corr_mat is not None:
            np.savez_compressed(os.path.join(path,'corr_mat.npy'), self.corr_mat)
        with open(os.path.join(path,'attrs.pkl'), 'wb') as f:
            pickle.dump(self,f)
        

    @classmethod
    def load(cls, path):
        with open(os.path.join(path,'attrs.pkl'), 'rb') as f:
            return pickle.load(f)


class SilhouetteTable:
    def __init__(self,silhouettes):
        self.silhouettes = silhouettes
        
    def __repr__(self):
        out = ''
        for s in self.silhouettes: 
            out += str(s) + "\n"
        return out 

    def _repr_html_(self):
        table = "<table>"
        table += "<tbody>"
        table += (
            "<tr>"
            '<th style="text-align: left"> Model </th>'
            '<th style="text-align: left"> n_clusters </th>'
            '<th style="text-align: left"> silhouette_score </th>'
            "</tr>"
        )
        for s in self.silhouettes:
            table += (
            "<tr>"
            '<td style="text-align: left"> %s </td>'
            '<td style="text-align: left"> %s </td>'
            '<td style="text-align: left"> %s </td>'
            "</tr>"
            ) %(s[0],str(s[1]), str(s[2]))
        return table