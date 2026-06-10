# 垃圾短信识别：基于监督学习的分类应用设计与实现

本项目用于人工智能课程期末大作业，任务是使用监督学习完成 SMS 垃圾短信二分类。主模型为 `TF-IDF + PyTorch MLP`，并与多数类、随机预测和 Logistic Regression baseline 对比。

## 项目概述

项目围绕短信文本分类任务展开，主要包含数据整理、特征表示、模型训练、基线对比、参数敏感性分析和结果可视化等部分。实验报告和关键图表已整理在仓库中，便于查看整体实现过程和实验结论。

## 方法

- 使用 TF-IDF 表示短信文本。
- 使用 PyTorch 实现多层感知机分类模型。
- 设置多数类、随机预测、传统机器学习模型等 baseline 进行对比。
- 通过学习率、Dropout、隐藏层结构、TF-IDF 特征规模等实验分析模型表现。

## 目录说明

- 源码：数据处理、模型定义、训练评估和绘图逻辑。
- 配置：实验参数和默认设置。
- 数据：原始数据及处理后的训练、验证、测试划分。
- 结果：模型指标、对比实验和敏感性分析记录。
- 图表：实验报告中使用的可视化结果。
- 报告：课程作业报告的 Word 和 PDF 文件。
- 文档：课程要求和项目检查材料。

## 数据来源

数据集：UCI Machine Learning Repository, SMS Spam Collection。

引用：Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
