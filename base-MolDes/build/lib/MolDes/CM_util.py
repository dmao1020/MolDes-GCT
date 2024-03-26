import numpy as np
import math

file_directory_path = os.getcwd()+"/"
qm9_data_dir = file_directory_path+ "data_files/"

# loading a dictionary for QM9 data files
# "SMILES: dsgdb9nsd_xxxxxx.xyz
qm9_file_ls = np.load("%sQM9_file_name_wt_SMILES.npy"%(qm9_data_dir),allow_pickle="TRUE")[()]


element_dict = {"Co" : 27,
                "Si" : 14,
                "O" : 8,
                "H" : 1,
                "F" : 9,
                "C" : 6,
                "Cu" : 29,
                "N" : 7,
                "P" : 15,
                "S" : 16,
                "Cl" : 17,
                "Zr" : 40,
                "Ti" : 22,
                "Sn" : 50,
                "Na" : 11,
                "Pt" : 78}


"""
A general python code that calculates Coulomb Matrix from cartesian coordinate

"""
class CM_util:
    def __init__(self, args, H_remove = True):
        self.args =args
        self.H_remove = H_remove
        
    def create_CM_from_smi(smi):

        smi_fn = "%s%s"%(qm9_xyz_dir,qm9_file_ls[smi])
        #print ("smi_fn",smi_fn)
        CM_arr = CM_util.xyz_gen_CM(smi_fn)
        return CM_arr

    
"""
A python code that calculates Coulomb Matrix for QM9 dataset

"""
class qm9_CM_util:
    def __init__(self, args, H_remove = True):
        self.args =args
        self.H_remove = H_remove
    
    def create_CM_from_smi(smi):

        smi_fn = "%s%s"%(qm9_xyz_dir,qm9_file_ls[smi])
        #print ("smi_fn",smi_fn)
        CM_arr = CM_util.xyz_gen_CM(smi_fn)
        return CM_arr