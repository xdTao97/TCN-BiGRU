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
(1) First run preprocess.py to process the data set. Taking the SMAP, MLS and SMD datasets.

(2) Then, run main.py again to get the model running results.


## Notice
We can set parameters such as data set, multi-head graph attention and sliding window through the configuration file args.py, and the final running results will be saved in the output folder.





     
