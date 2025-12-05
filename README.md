# CLIPTTA [NeurIPS 2025]
Official implementation of the NeurIPS 2025 paper: CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation. Please followe the following instructions to execute the main experiments found in the paper.

## Install the library

The source code works as a library for Test-Time Adaptation of VLMs such as CLIP. Several methods from the state-of-the-art are ready to use off-the-shelf. First, install the library in your environment 

```
pip install -U pip
pip install -e .
```

Once installed, all the sub-packages will be available to use.

## Running scripts

The hyperparameters of each method are described in detail in the file ```configuration.py```, and the main file to execute any of these 
adaptation strategies is ```main.py```. We also include scripts for closed-set and open-set adaptation inside the folder ```scripts```, 
with the necessary configurations to run CLIPTTA and competitor baselines. 

Key changes to be made on the scripts include adding your specific Python environment, as well as defining the parameters ```--root``` 
for the path where you store the project, ```--dataroot``` for the path where you store the datasets, and ```--save_root``` as the path 
where some results can be stored.
