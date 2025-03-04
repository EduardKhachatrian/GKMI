# GKMI / Automatic and Unsupervised Information Selection Method for Multimodal Remote Sensing

GKMI is a flexible, accurate, efficient, and interpretable attribute selection method that is based on information theory metrics and Graph Laplacians and allows determining the most informative and relevant attributes in heterogeneous datasets. The proposed approach assesses relevant information on a global and local level using two metrics that combine data structure and information content.  It selects relevant attributes for different regions of an image according to their physical characteristics and observation conditions. 

![Alt text](flowchart.jpg)

# $\color{lightblue}{\textbf{Installation :}}$

### $\color{olive}{\textbf{Clone the Repository}}$
 

```
git clone git@github.com:EduardKhachatrian/GKMI.git
cd GKMI
```

### $\color{olive}{\textbf{Create Environment Using .yml File}}$


```
conda env create -f environment.yml
conda activate gkmi
```
### $\color{olive}{\textbf{Build Install the GKMI Package}}$


```
python3 setup.py build install
```

### $\color{olive}{\textbf{Run the Proposed Scheme}}$
* $<\text{dataset}>$ corresponds to the dataset the user wants to apply for further experiments.

In ```sripts/*_config.py``` relpace ``` data_dir = 'test_datasets/ ``` with your local data path. Additionally, the user can tune the experiment's parameters for information selection and classification in config files.
    
```
python3 main.py -d <dataset>
```

# $\color{mediumpurple}{\textbf{Datasets :}}$

As an example we are using a few publicly available hyperspectral scenes. 

Scenes can be found here - [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)


### $\color{teal}{\textbf{Folder Structure of Input Data:}}$
    
        test_data
        │     
        └───PaviaU.mat
            PaviaU_gt.mat
            KSC.mat
            KSC_gt.mat
            Salinas.mat
            Salinas_gt.mat

### $\color{teal}{\textbf{Folder Structure of Output Data:}}$

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

# $\color{coral}{\textbf{Files :}}$

```gkmi/gkmi.py``` contains the information selection method.

```gkmi/performance_analysis.py``` contains the classification and performance evaluation part.

# $\color{cornflowerblue}{\textbf{References :}}$

* E. Khachatrian, S. Chlaily, T. Eltoft and A. Marinoni, "A Multimodal Feature Selection Method for Remote Sensing Data Analysis Based on Double Graph Laplacian Diagonalization," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 11546-11566, 2021, doi: 10.1109/JSTARS.2021.3124308.

* E. Khachatrian, S. Chlaily, T. Eltoft, W. Dierking, F. Dinessen and A. Marinoni, "Automatic Selection of Relevant Attributes for Multi-Sensor Remote Sensing Analysis: A Case Study on Sea Ice Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 9025-9037, 2021, doi: 10.1109/JSTARS.2021.3099398.

* E. Khachatrian, S. Chlaily, T. Eltoft, P. Gamba and A. Marinoni, "Unsupervised Band Selection for Hyperspectral Datasets by Double Graph Laplacian Diagonalization," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, 2021, pp. 4007-4010, doi: 10.1109/IGARSS47720.2021.9553127.


