# Anomaly Detection for Multivariate Time Series Data Based on Multi-Head Graph Attention


## Datasets
数据集下载地址：
https://github.com/Lliang97/Spacecraft-Anonamly-Detection/tree/main/datasets/data
下载数据集并导入到项目中，效果展示如下：
![dataset](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/dataset.png)
## Installation
```
1、安装python。
2、使用git命令下载项目到本地： http://10.12.52.24/taoxiaodong/timeseriesanomalydetect.git。也可以直接点击右上角下载按钮下载该项目。
3、安装conda。
4、根据environments.txt安装相应的运行环境。
```

## Running code
* 1、先运行preprocess.py,对数据集进行处理。以SMAP数据集为例，运行结果如下：
![datasetProcess](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/datasetProcess.png)
* 2、再运行main.py，得到模型运行结果。
![output_example](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/output_example.png)

## Notice
我们可以通过配置文件args.py进行数据集、多头图注意力以及滑动窗口等参数的设置，最后的运行结果都会保存的output文件夹中。
如下图所示：
![result](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/output.png)


## Experiment
模型已经确定，所有实验均可通过配置文件args.py修改参数进行。
* （1）通过修改args.dataset可以得到基础的实验结果。两组数据集分别进行十次实验，求取平均值，获得最终结果。以SMAP数据集为例，获取结果如图ex_result所示：![ex_result](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/ex_result.png)
图中bf_result是我们采用的最终结果，统计其中的f1，precision，recall得到TABLE II实验结果，统计AUC指标得到Fig4和Fig5关于AUC的实验结果。
* （2）统计并计算summary.txt中的epsilon_result和pot_result得到TABLE IV的实验结果。
* （3）在args.py文件中修改num_heads为1,2,3,4得到Fig6的实验结果。
* （4）修改args.py文件中的lookback值，设置为30,60,60,110,130得到Fig7的实验结果。
* （5）在TABLE III中，我们进行了Ablation Study。通过在mtad_gat.py文件中，注释相应的模块，得到最后的实验结果。注释模块参考下图image_annotation：
![image_annotation](http://10.12.52.24/taoxiaodong/timeseriesanomalydetect/-/blob/main/fig/mtad_gat.png)。
同时将modules.py中的aggregation类中的代码修改为如下所示：
```
    '''
    #使用x,h_feat,h_temp三个参数
    def forward(self, x1, x2, x3):
        x1_1 = x1
        x1_2 = x2
        x1_3 = x3

        x2_1 = torch.cat((x1_1, x1_2), 2)
        x2_1 = self.conv_concat2(x2_1)

        x2_2 = torch.cat((x1_1, x1_3), 2)
        x2_2 = self.conv_concat3(x2_2)

        x3_1 = torch.cat((x2_1, x2_2), 2)
        x3_1 = self.conv5(x3_1)
        x3_1 = x3_1.transpose(1, 2)
        return x3_1
    '''
    #消融实验中只使用h_temp或者只使用h_feat
    def forward(self, x1, x2):
        x1_1 = x1
        x1_2 = x2

        x2_1 = torch.cat((x1_1, x1_2), 2)
        x2_1 = self.conv_concat2(x2_1)

        x3_1 = self.conv5(x2_1)
        x3_1 = x3_1.transpose(1, 2)
        return x3_1
```
    