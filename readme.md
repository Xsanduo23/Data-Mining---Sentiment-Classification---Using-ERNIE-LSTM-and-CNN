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
