# Import python packages
import math
import time
import random
import os

import numpy as np
from numpy import linalg as LA

# rdkit library and functions
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from IPython.display import display
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles

class read_qm9_xyz_files:
    def __init__(self, args, H_remove = True):
        self.args =args
        self.H_remove = H_remove

    ## molecular information ##
    def create_dsgdb9nds_fn(self):
        args = self.args
        idx = args["xyz_idx"]
        idx_len = len(str(idx))
        fn = "dsgdb9nsd_"
        for i in range(6-idx_len):
            fn += "0"
        fn += "%s.xyz"%idx
        return fn
    def load_xyz_file_name(self):
        args = self.args
        file_dir = args["xyz_dir"]
        #read_xyz_util = read_xyz_files(args)
        fn = self.create_dsgdb9nds_fn()
        #print (file_dir+fn)
        return file_dir+fn
    
    def read_file(self):
        args = self.args
        #read_xyz_util = read_xyz_files(args)
        fn = self.load_xyz_file_name()

        f = open(fn)
        # for line in f:
        #     print (line)
        
    
    def file_read_xyz_coord(self):
        #print ("hello!")
        args = self.args
        H_remove = self.H_remove
        #read_xyz_util = read_xyz_files(args)
        fn = self.load_xyz_file_name() #read_xyz_util.load_xyz_file_name(args)

        f = open(fn)
        n_atoms = int(f.readline())
        #print ("n_atoms:", n_atoms)
        gdb_line = f.readline().split()
        xyz_arr = np.zeros((n_atoms, 4))
        for i in range(n_atoms):
            line = f.readline().split()
            atom_i_type_idx = float(atom_dict[line[0]])
            xyz_arr[i,:] = [atom_i_type_idx, float(line[1]), float(line[2]), float(line[3])]
        #print (xyz_arr)
        if H_remove == True:
            H_idx = np.where(xyz_arr[:,0]==1)[0]
            if len(H_idx)!=0:
                new_xyz_arr = np.delete(xyz_arr, H_idx, axis=0)
            else:
                new_xyz_arr = xyz_arr
        else:
            new_xyz_arr = xyz_arr

        return new_xyz_arr, gdb_line
    
    def file_read(self, return_var_name):
        args = self.args
        #read_xyz_util = read_xyz_files(args)
        fn = self.load_xyz_file_name()

        f = open(fn)
        n_atoms = int(f.readline())
        gdb_line = f.readline().split()
        for i in range(n_atoms):
            f.readline()
        energy_line = f.readline().split()
        smi0, smi1 = f.readline().split()
        # print ("smi0:",smi0)
        # print ("smi1:",smi1)
        inchi0, inchi1 = f.readline().split()

        if return_var_name == "smi":
            return smi0, gdb_line
        elif return_var_name =="smi_canonical":
            return smi1, gdb_line
        if return_var_name == "inchi":
            return inchi0, gdb_line

class GCT_util:
    def __init__(self, args, x = np.linspace(-300, 300, 100), cum_pdf_norm_stat = True, var_power = 1, Aij_power = 1):
        self.args = args
        self.x = x # the list of input value x for f(x), where f(x) is the gershgorin circle PDF
        self.cum_pdf_norm_stat = cum_pdf_norm_stat # weather the CDF is normalized
        self.var_power = var_power # The power value tau on the standard deviation of f_i
        self.Aij_power = Aij_power # The power value on Mij
    
    ### nomralized probability density function (PDF) ###
    def normalPdf(self, mean, variance):
        x = self.x
        return (1/np.sqrt(2*np.pi*variance))*np.exp(-0.5*(x-mean)**2/variance)
    
    ### nomralized cumulative distribution function (CDF) ###
    def normalCdf(self, mean, variance):
        x = self.x
        pdf = normalPdf(x, mean, variance)
        return np.cumsum(pdf)/np.sum(pdf)
    
    def mu_var_ls(CM_arr):
    count = 0
    mu_var_ls = []
    for i in range(np.shape(CM_arr)[0]):
        #print (i)
        count +=1
        row_i = CM_arr[i,:]
        Aii = row_i[i]
        #print (Aii)
        sum_Aij = np.sum(row_i)-Aii
        mu_var_ls.append([Aii, np.abs(sum_Aij)])
        #print (sum_Aij)
    return count, np.array(mu_var_ls)
    
    ### nomralized cumulative distribution function (CDF) ###
    def smi_to_cum_pdf_calc(self, smi):
        cum_pdf_norm_stat_ = self.cum_pdf_norm_stat
        if cum_pdf_norm_stat_ == True:
            cum_pdf_norm_stat = 0
        else:
            cum_pdf_norm_stat = 1
        var_power = self.var_power
        Aij_power = self.Aij_power
        x = self.x
        CM = np.array(create_CM(smi))
        count, mu_var_arr = self.mu_var_ls(CM)
        d = 1/count

        cum_pdf = (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_var_arr[0,0], mu_var_arr[0,1])#**2)
        sq_cum_pdf = (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_var_arr[0,0], mu_var_arr[0,1]**var_power)
        for pdf_i in range(len(mu_var_arr)-1):
            mu_i, var_i = mu_var_arr[pdf_i+1,:]
            cum_pdf += (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_i, var_i)#**2)
            sq_cum_pdf += (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_i, var_i**var_power)
        return cum_pdf, sq_cum_pdf
    def cum_pdf_auc():
        auc_ls = []
        for i in progressbar(range(len(smi_ls)), "Computing: ", 40):
            #print ("Molecule number: %s"%(i))
            smi_i = smi_ls[i]
            dir_fn = "%sgersh_%s.png"%(GPBO_gersh_savefig_dir, smi_i)

            # calculate eigenvalues of the coulomb matrix
            smi_CM_arr = np.array(create_CM(smi_i))
            #print ("smi_CM_arr:",smi_CM_arr)
            w, v = LA.eig(smi_CM_arr)
            count, mu_var_arr = mu_var_ls(smi_CM_arr)


            x = np.linspace(-300, 300, 100)
            d = 1/count

            cum_pdf = (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_var_arr[0,0], mu_var_arr[0,1])#**2)
            for pdf_i in range(len(mu_var_arr)-1):
                mu_i, var_i = mu_var_arr[pdf_i+1,:]
                cum_pdf += (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_i, var_i)#**2)

            auc_1_i = np.trapz(cum_pdf, dx=1)
            auc_5_i = np.trapz(cum_pdf, dx=5)
            #print ("Area under curve:", auc_5_i )
            #auc_ls.append(auc(x,cum_pdf))
            auc_ls.append([auc_1_i, auc_5_i, w])
        return np.array(auc_ls)
    def auc_calc(smi, cum_pdf_norm_stat):
        smi_CM_arr = np.array(create_CM(smi))
        count, mu_var_arr = mu_var_ls(smi_CM_arr)


        x = np.linspace(-300, 300, 100)
        d = 1/count

        cum_pdf = (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_var_arr[0,0], mu_var_arr[0,1]**2)
        for pdf_i in range(len(mu_var_arr)-1):
            mu_i, var_i = mu_var_arr[pdf_i+1,:]
            cum_pdf += (d/d**cum_pdf_norm_stat)*normalPdf(x, mu_i, var_i**2)

        auc_1_i = np.trapz(cum_pdf, dx=1)
        auc_5_i = np.trapz(cum_pdf, dx=5)
        return auc_1_i ,auc_5_i
