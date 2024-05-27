# 环境
```shell
jieba~=0.42.1
numpy~=1.21.5
matplotlib~=3.5.1
gensim~=4.2.0
tqdm~=4.64.1

scikit-learn~=1.0.2
boto3~=1.24.77
requests~=2.28.1
botocore~=1.27.77
regex~=2022.7.9
tensorboardx~=2.5.1
python版本建议3.7或3.9

```

# 目录

run.py 为运行主程序
train_eval 为训练函数
utils 为预处理函数
Datas 里存档训练集
pretrain 里存放bert预训练文件
pytorch_pretrained里为优化器等库包，为防止下载版本错误所以放在本地
models里 为模型


# 运行
```shell
python run.py
```


# 问题
模型早停后，只出现第一张图
# 方法
叉掉窗口，让程序继续运行，以得到后续图


 # 问题
报缺sklearn 
# 解决
安装scikit-learn库包 


 # 问题
模型短时间只输出第一轮结果
# 解决
不是代码停了，而是运行的很慢。因为python环境里装的是cpu的pytorch，需要gpu显卡版的pytorch

# 问题
import pyarrow.lib as _lib
ImportError: DLL load failed while importing lib: 找不到指定的程序
解决方法，更新pyarrow
```shell
pip install --user --upgrade pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
```
