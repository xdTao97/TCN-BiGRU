import numpy as np
import torch
# 导入Matplotlib库
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import SlidingWindowDataset


# 定义函数计算所有测试集样本的特征重要性并返回平均重要性
def feature_importance_all_samples(model, test_loader, input_dim):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    feature_importance_sum = torch.zeros(input_dim)  # 初始化特征重要性总和

    feature_importance_sum = feature_importance_sum.to(device)

    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)
            # 前向传播获取原始输出
            original_output, recon = model(data)
            # 针对每个特征逐个进行置零操作并重新计算输出
            for i in range(data.size(2)):  # 遍历特征维度
                # 复制输入数据
                modified_data = data.clone()
                # 将当前特征置零
                modified_data[:, i] = 0
                # 通过模型进行前向传播
                modified_output, recon = model(modified_data)
                # 计算输出变化
                output_change = torch.abs(original_output - modified_output)
                # 计算特征重要性（输出变化越大，特征重要性越高）
                # 累加特征重要性
                feature_importance_sum[i] += torch.mean(output_change)

    # 计算平均特征重要性
    feature_importance_avg = feature_importance_sum / len(test_loader) + 0.5  # 注意  调整 在这里！！！！

    return feature_importance_avg


# 可视化特征重要性
# 创建柱状图
def featureimportance_all_samples(model, test_loader, input_dim):
    # 计算所有样本的特征重要性
    importance_all_samples = feature_importance_all_samples(model, test_loader, input_dim)
    importance_all_samples = importance_all_samples.cpu().numpy()
    # 计算特征重要性的总和
    total_importance = sum(importance_all_samples)
    # 对特征重要性进行归一化处理
    normalized_importance = [imp / total_importance for imp in importance_all_samples]
    print(normalized_importance)
    # 生成示例颜色列表，可以根据需要自定义
    # colors = ['dodgerblue', 'darkorange', 'seagreen', 'crimson', 'darkorchid', 'royalblue', 'g']
    print("33333333333333")
    # 特征名称列表
    feature_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y']
    plt.figure(figsize=(10, 5), dpi=100)
    bars = plt.barh(range(input_dim), normalized_importance)
    # plt.xlabel('特征贡献度')
    plt.xlabel('A')
    # plt.ylabel('特征')
    plt.ylabel('B')
    # plt.title('所有样本特征重要性分析')
    plt.title('C')

    # 设置特征名称作为纵轴刻度和标签
    plt.yticks(range(input_dim), feature_names)
    plt.savefig('./fig/feature.png')
    # 显示柱状图并设置每个柱子的颜色
    # for bar, color in zip(bars, colors):
    #     bar.set_color(color)
    plt.show()
