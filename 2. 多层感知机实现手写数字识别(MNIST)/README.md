## 一、 背景介绍
MNIST 手写数字数据集是机器学习领域中广泛使用的图像分类数据集。它包含 60,000 个训练样本和 10,000 个测试样本。这些数字已进行尺寸规格化,并在固定尺寸的图像中居中。每个样本都是一个 784×1 的矩阵,是从原始的 28×28灰度图像转换而来的。MNIST 中的数字范围是 0 到 9。下面显示了一些示例。 

![示例图片](../1.%20Softmax实现手写数字识别(MNIST)/img/mnist.png)

## 二、实验目的
构建自己的多层感知机，实现MNIST 手写数字识别
-  实现SGD优化器 (`./optimizer.py`)
-  实现全连接层FCLayer前向和后向计算 (`layers/fc_layer.py`)
-  实现激活层SigmoidLayer前向和后向计算 (`layers/sigmoid_layer.py`)
-  实现激活层ReLULayer前向和后向计算 (`layers/relu_layer.py`)
-  实现损失层EuclideanLossLayer (`criterion/euclidean_loss.py`)
-  实现损失层SoftmaxCrossEntropyLossLayer (`criterion/softmax_cross_entropy.py`)

## 三、评价指标
* Accuracy 准确率: 分类正确的样本数除以总样本数。

## 四、实验过程
见各个训练文件
* train.ipynb                       本文件    
* mlp_1 momentum.ipynb              训练过程，尝试加入动量  
* mlp_2 learning_rate_SGD.ipynb     训练过程，尝试不同的学习率  
* mlp_3 weight_decay.ipynb          训练过程，尝试不同的权重衰减率  
* mlp_4 batch_size.ipynb            训练过程，尝试不同的正则化系数  

## 五、结果汇总：
### 5.1 单层隐含层的感知机
| model | batch size | learning rate SGD | momentum | weight decay | acc validate | acc test |
|----------|----------|----------|---------|----------|----------|----------|
| Euclidean+Sigmoid     | 10 | 0.001 | 0.55 | 0.0001    | 0.9046 | 0.9161 | 
| Euclidean+ReLU        | 10 | 0.001 | 0.99 | 0.0001    | 0.9678 | 0.9661 | 
| CrossEntropy+Sigmoid  | 10 | 0.001 | 0.55 | 0.00001   | 0.9257 | 0.9267 | 
| CrossEntropy+ReLU     | 10 | 0.001 | 0.99 | 0.00001   | 0.9786 | 0.9802 | 
*  <font color="green">注:batch size-批大小; learning rate SGD-学习率;momentum-动量;weight decay-权重衰减率。</font>

### 5.2 具有多层隐含层的多层感知机

| model | hid_layers | neurons | train time | acc validate | acc test |
|----------|----------|----------|---------|----------|----------|
| CrossEntropy+ReLU | 1 | 128               | 321.255   | 0.9799 | 0.9812 | 
| CrossEntropy+ReLU | 2 | 512,128           | 997.467   | 0.9841 | 0.9851 | 
| CrossEntropy+ReLU | 2 | 256,64            | 552.641   | 0.9801 | 0.9831 | 
| CrossEntropy+ReLU | 2 | 300,100           | 640.250   | 0.9816 | 0.9829 | 
| CrossEntropy+ReLU | 3 | 512,256,128       | 1326.537  | 0.9801 | 0.9818 | 
| CrossEntropy+ReLU | 4 | 512,256,128,64    | 1421.584  | 0.9802 | 0.9833 | 
*  <font color="green">注:hid_layers-隐含层的个数; neurons-全连接层神经元的数量;train time-模型训练共花费时间。单位(秒)。</font>

 ### 5.3 <font color="green">其它过程和结果具体见各个ipynb训练文件</font>

***
***
## 环境参考：

| model | version |
|----------|----------|
| python                    | 3.10.13 |
| matplotlib                | 3.4.3 |
| numpy                     | 1.22.3 |
| pandas                    | 1.5.3 |

## 文件清单如下:
| file/folder | remark |
|----------|----------|
|   mlp.ipynb                       | 训练过程 |
|   mlp_1 momentum.ipynb            | 训练过程，尝试加入动量 |
|   mlp_2 learning_rate_SGD.ipynb   | 训练过程，尝试不同的学习率 |
|   mlp_3 weight_decay.ipynb        | 训练过程，尝试 |
|   mlp_4 batch_size.ipynb          | 训练过程，尝试不同的正则化系数 |
|   criterion/                      | 损失函数 |
|   layers/                         | 隐含层（激活函数） |
|   network.py                      | 定义了网络,包括其前向和反向计算 |
|   optimizer.py                    | 定义了随机梯度下降(SGD),用于完成反向传播和参数更新 |
|   solver.py                       | 定义了训练和测试过程需要用到的函数 |
|   plot.py                         | 用来绘制损失函数和准确率的曲线图 |
