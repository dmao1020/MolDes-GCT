import numpy as np
import math
import os
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
def create_CM_from_smi(xyz_dir, smi):
    smi_fn = "%s%s"%(xyz_dir,qm9_file_ls[smi])
    # print ("smi_fn",smi_fn)
    CM_arr = xyz_gen_CM(smi_fn)
    # print ("CM_arr:",CM_arr)
    return CM_arr

def cm(atom_symbol,xyz_coord,max_n_atoms):
    full_mat = np.zeros((max_n_atoms,max_n_atoms))
    #print (full_mat)
    n_atoms = np.shape(atom_symbol)[0]
    #print(n_atoms)
    cm =[]
    for i in range(n_atoms):
        row = []
        atom_i = str(atom_symbol[i])
        #print ("atom_i",atom_i)
        Z_i = element_dict[atom_i]
        #print ("Z_i",Z_i)
        R_i = np.array(xyz_coord[i,:])
        
        for j in range(n_atoms):
            atom_j =str(atom_symbol[j])
            Z_j = element_dict[atom_j]
            #print ("atom_j",atom_j)
            #print ("Z_j",Z_j)
            R_j =np.array(xyz_coord[j,:])
            #print (R_i,R_j)
            
            if i == j:
                cii = 0.5*Z_i**2.4
                #print ("cii",cii)
                row.append(cii)
            else:
                #print ("R_i,R_j",R_i,R_j)
                R_ij=R_i-R_j
                #print ("R_ij",R_ij)
                R_ij_norm= np.linalg.norm(R_ij)
                #print ("R_ij_norm",R_ij_norm)
                cij = (Z_i*Z_j)/(R_ij_norm)
                #print ("cij",cij)
                row.append(cij)
                
        #print ("row",np.shape(row))
        row = np.array(row)
        cm.append(row)
    full_mat[:n_atoms,:n_atoms] = cm
    return np.array(full_mat)
def cm_mu_var_ls(CM_arr):
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

def read_xyz(fn):
    xyz = open(fn,"r")
    # print (fn)
    line = xyz.readline()
    n_of_atoms = str(int(line))
    # print ("n_of_atoms",n_of_atoms)
    tag_j,index_j, A_j,B_j,C_j, mu_j, alpha_j, homo_j,lumo_j, gap_j, r2_j,zpve_j,U0_j,U_j, H_j, G_j,Cv_j=xyz.readline().split()    
    xyz_coord = []
    
    for atom in range(int(n_of_atoms)):
        line = xyz.readline().split()
        xyz_coord.append(line)
    xyz_coord = np.array(xyz_coord)
    #print (xyz_coord)
    xyz_coord_float = xyz_coord[:,1:-1]
    for i in range(len(xyz_coord_float)):
        for j in range(3):
            xyz_coord_ij = xyz_coord_float[i,j]
            if xyz_coord_ij.find("*^") != -1:
                #print (xyz_coord_ij)
                ab_idx = xyz_coord_ij.find("*^")
                power_val = float(xyz_coord_ij[-1])
                xyz_coord[i,j+1] = float(xyz_coord_ij[:ab_idx])*10**(-power_val)
            else:
                xyz_coord[i,j+1] = float(xyz_coord_float[i,j])
    return xyz_coord
    
    
def xyz_gen_CM(fn):
    xyz = open(fn,"r")
    #print (fn)
    line = xyz.readline()
    n_of_atoms = str(int(line))
    #print ("n_of_atoms",n_of_atoms)
    tag_j,index_j, A_j,B_j,C_j, mu_j, alpha_j, homo_j,lumo_j, gap_j, r2_j,zpve_j,U0_j,U_j, H_j, G_j,Cv_j=xyz.readline().split()    
    xyz_coord = []
    
    for atom in range(int(n_of_atoms)):
        line = xyz.readline().split()
        xyz_coord.append(line)
    xyz_coord = np.array(xyz_coord)
    #print (xyz_coord)
    xyz_coord_float = xyz_coord[:,1:-1]
    for i in range(len(xyz_coord_float)):
        for j in range(3):
            xyz_coord_ij = xyz_coord_float[i,j]
            if xyz_coord_ij.find("*^") != -1:
                #print (xyz_coord_ij)
                ab_idx = xyz_coord_ij.find("*^")
                power_val = float(xyz_coord_ij[-1])
                xyz_coord[i,j+1] = float(xyz_coord_ij[:ab_idx])*10**(-power_val)
            else:
                xyz_coord[i,j+1] = float(xyz_coord_float[i,j])
                
                
    
    
    CM_arr = cm(xyz_coord[:,0],xyz_coord[:,1:-1].astype(float),int(n_of_atoms))
    return np.array(CM_arr)