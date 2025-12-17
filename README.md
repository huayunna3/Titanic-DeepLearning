# 🚢 Titanic Survival Prediction with Deep Learning

一个使用 **PyTorch** 构建的深度学习模型，用于预测泰坦尼克号乘客的生存情况。0.9425的成绩。

## 📊 项目概述

本项目目标是构建一个二分类模型，根据乘客的票务、姓名、舱位等特征，预测其在泰坦尼克号海难中的生存(`Survived=1`)或遇难(`Survived=0`)。

**核心成果**:
*   **模型性能**：在保留的验证集上取得了 **85.5%** 的准确率和 **0.836** 的F1分数。
*   **技术实践**：完整实现了数据清洗、特征工程、神经网络构建、动态训练调优及结果分析的全流程。

### 1. 特征工程
*   **真实票价计算**：观察样本数据发现，票号存在相同现象即有人拼团买票。
*   提取真实票价信息：根据`Ticket`字段，识别同行组，将总票价`Fare`均摊，更准确地反映个人支付能力。
  data['Fare']=data['Fare']/data.groupby('Ticket')['Ticket'].transform('count')
*   **年龄填补**：在年龄空白处有`Mr.`, `Mrs.`, `Miss`, `Master`，分别代表成年男性，已婚女性，小姐和小孩哥。
    df.loc[df['Name'].str.contains('xx') =年龄
*   **亲友团调整**：将`SibSp`和`Parch`(父母/子女)全部＋1，表示“乘客本人”，这样就没有零了。
*   **船舱等级**：提取`Cabin`首字母作为信息，替换。
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'X')
    df['Cabin'].unique()
    一共就9个单词，map替换一下
### 2. 训练策略
*   **优化器切换**：在训练后期（损失稳定后），将优化器从 **Adam** 切换为 **SGD**，降低学习率(`1e-3` -> `1e-5`)。
*   **模型结构**：`ELU`激活函数、`BatchNorm1d`批归一化层，并预留了`Dropout`层接口以应对过拟合。(关掉了，开到0.9模型训练200次还能预测到0.8以上，非常逆天）
*   **评估**：除准确率外，可视化**精确度、召回率、F1分数**，多维度评估模型性能。（情绪价值拉满）

### 3. 严谨的模型验证与分析
*   **多维度测试**：额外构建了“全女性存活/全男性死亡”的极端案例测试集和“全男性”样本集，深入验证模型在不同子群体上的表现与偏差。
*   **完整流程闭环**：代码包含最终对Kaggle测试集的预测与结果文件(`结果.csv`)生成功能。
*   **模型持久化**：训练完成后自动保存最优模型参数(`模型.pth`)。

## 🛠️ 技术栈

*   **框架**：PyTorch
*   **编程语言**：Python
*   **数据处理**：Pandas, NumPy, Scikit-learn (`StandardScaler`, `train_test_split`)
*   **可视化**：Matplotlib
*   **评估指标**：Accuracy, Precision, Recall, F1-Score (from `sklearn.metrics`)

## 📁 项目结构

```
titanic-deeplearning/
│
├── titanic-deeplearning.ipynb          # 主代码文件 (Jupyter Notebook / Kaggle Kernel)
├── README.md                           # 本说明文件
│
├── data/                               # 数据目录 (需从Kaggle下载)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── models/                             # 存放训练好的模型 (运行后生成)
│   └── 模型.pth
│
└── outputs/                            # 存放预测结果 (运行后生成)
    └── 结果.csv
```

## 🚀 快速开始

### 环境配置
1.  确保已安装 Python 3.7+。
2.  安装依赖包：
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib
    ```

### 运行步骤
1.  **获取数据**：从 [Kaggle Titanic 竞赛页面](https://www.kaggle.com/c/titanic/data) 下载 `train.csv` 和 `test.csv`，放置于 `data/` 目录下。
2.  **运行代码**：直接运行 `titanic-deeplearning.ipynb` 中的所有单元格。
3.  **查看结果**：预测结果将保存在 `outputs/结果.csv` 中，可用于提交至Kaggle。模型权重将保存在 `models/模型.pth`。

 ## 📈 分析与总结

训练过程稳定，损失持续下降，未见明显过拟合：
*   **验证集**：准确率 **85.5%**, F1分数 **0.836**
*   **“女性全存/男性全亡”测试集**：准确率 **87.6%**，证明模型成功捕捉到了数据中“妇孺优先”的强规则。
*   **“全男性”子集**：准确率约 **82.4%**，说明模型在去除了最强特征（性别）后，依然能依靠其他特征（舱位、票价等）进行有效预测。
*   但分数不高，也许是特征提取不到位，数据划分问题。
*   训练集和测试集难划分，模型受随机数影响极大，随机数种子16和66模型准确率能差3个点以上
## 💡 总结与展望

1.  **特征工程**：可尝试提取`Name`中的姓氏以更精确识别家庭
2.  **模型架构**：损失值一值很高，标签平滑后还在0.55以上，后续扩大模型深度并与机器学习结合。
3.  **提升有限**：啥也不干，把字符列全部删掉，年龄均值填充，男：1 女：2 分数也有0.74
4.  **模型倾向**：模型自信且偏向预测0，在预测test中进行了手动提高，后续查看特异性和真负例率
5.  **数据分布**：数据分布不均，可以试试不划分数据集，直接全部拿来训练。

## 👨‍💻 作者

**yunna hua**
*   Kaggle: [@yunnahua](https://www.kaggle.com/yunnahua)
*   GitHub: [huayunna3]
*   致力于将扎实的深度学习理论应用于解决实际问题。

## 📄 许可

本项目基于 Apache 2.0 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

---
**如果这个模型对你有帮助，现在这个版本会让你非常困惑**
