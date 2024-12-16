# 总述
**RAW2RGB** 是一个用于将 RAW 图像转换为 RGB 图像的深度学习模型。该项目的目标是通过神经网络将 RAW 图像转换为高质量的 RGB 图像，实现端到端的ISP的功能，替换掉传统ISP中的各个模块。

## 数据集准备
 
数据集1： ZRR数据集
数据集2： ISPW数据集
数据集3： xiaomi13ultra数据集： 下载链接：链接：https://pan.baidu.com/s/18qSbnlHHiGq3EW4Uv3cG5Q   提取码：9h9r 

### RAW 图像
准备一个包含 RAW 图像的数据集，支持的格式有 PNG（最好先转成PNG来存放高bit）, RAW 等。

### RGB 图像
每个 RAW 图像对应一个 RGB 图像（通常为 PNG、JPEG 格式），用于监督学习。

### 数据集结构示例

```bash
/dataset
    /train
        /raw_images
            - image1.raw
            - image2.raw
            ...
        /rgb_images
            - image1.png
            - image2.png
            ...
    /test
        /raw_images
            - test1.raw
            - test2.raw
            ...
        /rgb_images
            - test1.png
            - test2.png
            ...


### 测试
     python test.py 
### 训练
    python train.py
