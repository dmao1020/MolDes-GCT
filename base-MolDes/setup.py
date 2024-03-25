from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'MolDes python package'
LONG_DESCRIPTION = 'MolDes python package to create descriptor based on Coulomb matrix (CM_util); Gershgorin circle theorem (GCT_util); and Graph convolution network (GCN_util)'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="MolDes", 
        version=VERSION,
        author="Yun-Wen Mao",
        author_email="<dmao1020@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'MolDes', 'Chemistry', 'Descriptor', 'low-dimensional'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)