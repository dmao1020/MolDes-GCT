# MolDes-GCT
Welcome to MolDes, a powerful toolset for generating low-dimensional descriptors for molecular structures, based on the methodologies outlined in the paper titled "Eﬃcient interpolation of molecular properties across chemical compound space with low-dimensional descriptors" authored by Mao et al. [1] This repository provides a Python implementation of the techniques described in the paper, empowering users to efficiently create meaningful representations of molecular data.

# Features
1. Coulomb Matrix Construction: MolDes facilitates the construction of Coulomb matrices for molecular structures. 
2. Gerschgorin Circle Theorem Application: Leveraging the Gerschgorin Circle Theorem, MolDes helps users analyze the eigenvalue spectrum of Coulomb matrices. This analysis aids in identifying key molecular properties and refining descriptor generation.
3. Graph Neural Network Integration (still under construction): MolDes includes a Graph Neural Network (GNN) module tailored specifically for molecular data. This module utilizes the structural information encoded in molecular graphs to enhance descriptor generation accuracy.
4. Descriptor Generation: MolDes offers functionalities to generate low-dimensional descriptors from Coulomb matrices and GNN embeddings. These descriptors encapsulate crucial molecular features, facilitating various downstream tasks such as molecular property prediction, similarity analysis, and molecular clustering.

# Usage
To utilize MolDes for molecular descriptor generation, users can follow these steps:

1. Install the required dependencies specified in the "requirements.txt" file. For users Anaconda users, a .yml file is also provided.
2. Follow the provided examples in the "Tutorial" folder or integrate MolDes into your workflow by importing the necessary modules.
3. Utilize the provided functions to construct Coulomb matrices, analyze eigenvalue spectra, generate GNN embeddings, and create low-dimensional descriptors for your molecular datasets.
4. Customize and extend the functionalities of MolDes to suit your specific research or application requirements.
5. For users who are interested in testing the code with QM9 dataset, please visit https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904 and download the dataset. [2]

# Reference
If you found this code useful, please consider citing the following work:

[1] Yun-Wen Mao, Roman V. Krems; Eﬃcient interpolation of molecular properties across chemical compound space with low-dimensional descriptors. Mach. learn.: sci. technol. 20 March 2024; https://doi.org/10.1088/2632-2153/ad360e

[2] Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014). Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5
