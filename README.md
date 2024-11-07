# Flatten_transformer_paddle

# 模型简介
FLatten Transformer是一种新型的视觉Transformer，它通过使用一种称为Focused Linear Attention的模块来解决传统自注意力机制中的计算复杂度问题。这种模块旨在提高效率和表达能力，同时保持较低的计算复杂度。Focused Linear Attention通过两个主要的改进来实现这一点：

1.FLatten Transformer分析了现有线性注意力方法在焦点能力方面的局限性，并提出了一种简单的映射函数来调整查询和键的特征方向，使得相似的查询-键对更加接近，而不相似的则相互排斥。这种方法使得注意力权重分布更加尖锐，有助于模型关注信息量更丰富的特征。

2.为了解决线性注意力中特征多样性不足的问题，FLatten Transformer引入了一个高效的秩恢复模块。这个模块通过在原始注意力矩阵上应用额外的深度卷积（Depthwise Convolution, DWC）来恢复特征多样性，从而保持不同位置输出特征的多样性。

## 使用方法
1.下载paddleclas，将FLATTENMODELS复制到paddleclass根目录下。

2.ppcls\configs\ImageNet中包含yaml配置文件的Flatten_transformer文件夹放入paddleclas对应位置（ppcls\configs\ImageNet）。


## 下载权重
权重文件链接：https://pan.baidu.com/s/1J99bqai_RBCxGEBFDEcZTw?pwd=1111, 提取码：1111 


## 参考
本项目代码参考自：https://paperswithcode.com/paper/flatten-transformer-vision-transformer-using
