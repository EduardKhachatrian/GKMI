# GKMI / Automatic and Unsupervised Information Selection Method for Multimodal Remote Sensing

GKMI is a flexible, accurate, efficient, and interpretable attribute selection method that is based on information theory metrics and Graph Laplacians and allows determining the most informative and relevant attributes in heterogeneous datasets. The proposed approach assesses relevant information on a global and local level using two metrics that combine data structure and information content.  It selects relevant attributes for different regions of an image according to their physical characteristics and observation conditions. 

![flowchart.jpg](attachment:flowchart.jpg)

# Installation :

$\color{green}{\textbf{Clone the Repository}}$
 
```
git clone git@github.com:EduardKhachatrian/GKMI.git
cd GKMI
```

$\color{green}{\textbf{Create Environment Using .yml File}}$
 
```
conda env create -f environment.yml
conda activate gkmi
```
$\color{green}{\textbf{Build Install the GKMI Package}}$
```
python3 setup.py build install
```

$\color{green}{\textbf{Run Information Selection and Classification}}$, <$\text{dataset}$> corresponds to the dataset the user wants to run the experiment. 

In ```sripts/*_config.py``` relpace ``` data_dir = 'test_datasets/ ``` with your local data path. Additionally, the user can tune the experiment's parameters for information selection and classification in config files.
    
```
python3 main.py -d <dataset>
```

# Datasets :

As an example we are using a few publicly available hyperspectral scenes. 

They can be found here - [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)


$\color{blue}{\textbf{Folder Structure of Input Data:}}$
    
        test_data
        │     
        └───PaviaU.mat
            PaviaU_gt.mat
            KSC.mat
            KSC_gt.mat
            Salinas.mat
            Salinas_gt.mat

$\color{blue}{\textbf{Folder Structure of Output Data:}}$

        classification_outputs
        │     
        └───Pavia  
        │   │     
        │   └───Pavia_classified.eps
        │       Pavia_results.eps
        │       gkmilogfile.log
        │
        └───KSC
        │   │
        │   └───KSC_classified.eps
        │       KSC_results.eps
        │       gkmilogfile.log
        │
        └───Salinas
            │
            └───Salinas_classified.eps
                Salinas_results.eps
                gkmilogfile.log 

# Files :

```src/gkmi.py``` contains the information selection method.

```src/performance_analysis.py``` contains the classification and performance evaluation part.

# References :

* E. Khachatrian, S. Chlaily, T. Eltoft and A. Marinoni, "A Multimodal Feature Selection Method for Remote Sensing Data Analysis Based on Double Graph Laplacian Diagonalization," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 11546-11566, 2021, doi: 10.1109/JSTARS.2021.3124308.

* E. Khachatrian, S. Chlaily, T. Eltoft, W. Dierking, F. Dinessen and A. Marinoni, "Automatic Selection of Relevant Attributes for Multi-Sensor Remote Sensing Analysis: A Case Study on Sea Ice Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 9025-9037, 2021, doi: 10.1109/JSTARS.2021.3099398.

