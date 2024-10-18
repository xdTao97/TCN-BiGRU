# Anomaly Detection for Multivariate Time Series Data Based on Multi-Head Graph Attention


## Datasets


The following datasets are kindly released by different institutions or schools. Raw datasets could be downloaded or applied from the link right behind the dataset names. The processed datasets can be found here (SMD, SMAP, and MSL).

Server Machine Datase (SMD) Download raw ![dataset](https://github.com/NetManAIOps/OmniAnomaly)

Collected from a large Internet company containing a 5-week-long monitoring KPIs of 28 machines. The meaning for each KPI could be found here.

Soil Moisture Active Passive satellite (SMAP) and Mars Science Laboratory rovel (MSL) Download raw ![dataset](https://github.com/Lliang97/Spacecraft-Anonamly-Detection/tree/main/datasets/data)

They are collected from the running spacecraft and contain a set of telemetry anomalies corresponding to actual spacecraft issues involving various subsystems and channel types.


## Installation
```
1. Install python.
2. Use git command to download the project to local.
3. Install condconda.
4. Install the corresponding operating environment according to environments.txt.
```

## Running code
(1) First run preprocess.py to process the data set. Taking the SMAP data set as an example, the running results are as follows: <br>
W
(2) Run main.py again to get the model running results.


## Notice
We can set parameters such as data set, multi-head graph attention and sliding window through the configuration file args.py, and the final running results will be saved in the output folder.


## Experiment
The model has been determined, and all experiments can be performed by modifying the parameters in the configuration file args.py.
* (1) Basic experimental results can be obtained by modifying args.dataset. The two sets of data sets were subjected to ten experiments respectively, and the average value was calculated to obtain the final result. We show the running results in the folder output.
The bf_result in the figure is the final result we used. Statistics on f1, precision, and recall get the TABLE II experimental results. Statistics on the AUC index are used to obtain the experimental results on AUC in Figures 4 and 5 in the paper.
* (2) Statistics and calculation of epsilon_result and pot_result in summary.txt to obtain the experimental results of TABLE IV.
* (3) Modify num_heads in the args.py file to 1, 2, 3, 4 to get the experimental results of Fig6.
* (4) Modify the lookback value in the args.py file and set it to 30, 60, 60, 110, 130 to get the experimental results of Fig7.
* (5) In TABLE III, we conducted Ablation Study. By annotating the corresponding modules in the mtad_gat.py file, the final experimental results are obtained. For the annotation module, please refer to the image_annotation image below:

W

     
