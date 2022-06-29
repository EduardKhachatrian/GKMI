# GKMI / Unsupervised Information Selection Method

GKMI is a flexible, accurate, efficient, and interpretable attribute selection method 
that is based on information theory metrics and Graph Laplacians and allows determining 
the most informative and relevant attributes in heterogeneous datasets. 

Authors :  Eduard Khachatrian    <eduard.khachatrian@uit.no>
           Saloua Chlaily        <saloua.chlaily@uit.no> 
           Andrea Marinoni       <andrea.marinoni@uit.no>


Clone the repository
```
git clone git@github.com:EduardKhachatrian/GKMI.git
cd GKMI
```

Create environment using .yml file
```
conda env create -f environment.yml
conda activate gkmi
```

Build install the gkmi package
```
python3 setup.py build install
```

Input Data:
    Where to download 
    Folder structure 
        
        test_data
        │     
        └───Pavia
            │     
            └───PaviaU_gt.mat

Run Classification and Feature Extraction, <dataset> corresponds to the dataset the user wants to run the experiment
    
In sripts/pavia_config.py relpace ``` data_dir = 'test\sssts\ ``` with your local data path. Also, the user can tune the experiment's parameters.
    
```
python3 main.py -d <dataset>
```

Output Data:
    skjvnkjvdnsv.log
    
    classification_output
        │     
        └───Pavia
            │     
            └───outputksvksvnc.mat 

