# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
# import pickle as pkl
import pandas as pd
import seaborn as sns
# from xarray.plot.utils import plt
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# 训练模型
def train(config, model, train_iter, dev_iter, test_iter):
    # 记录训练开始时间，用于计算训练时长
    start_time = time.time()
    # 初始化训练集和验证集的准确率和损失值列表，用于后续绘图分析
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    Iter_list = []

    # 将模型设置为训练模式
    model.train()
    # 获取模型的所有参数
    param_optimizer = list(model.named_parameters())
    # 定义不需要weight decay的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 组织模型参数，分别对需要和不需要weight decay的参数进行设置
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # 定义优化器，这里使用BertAdam优化器，并设置学习率和warmup策略
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # 初始化batch计数器，验证集最佳损失值，上次提升的batch数及提升标志
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    # 开始按照epoch进行训练
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # 按batch迭代数据
        for i, (trains, labels) in enumerate(train_iter):
            # 前向传播
            outputs = model(trains)
            # 清空梯度
            model.zero_grad()
            # 计算损失
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 每隔一定batch，评估一次模型在训练集和验证集上的表现
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 如果验证集损失下降，更新最佳损失值，保存模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                # 计算训练时间
                time_dif = get_time_dif(start_time)
                # 打印训练信息
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 记录训练集和验证集的准确率和损失值
                Iter_list.append(total_batch)
                train_acc_list.append(train_acc)
                train_loss_list.append(loss.item())
                val_acc_list.append(dev_acc)
                val_loss_list.append(dev_loss.clone().detach().cpu().numpy())
                # 重新设置模型为训练模式
                model.train()
            total_batch += 1
            # 如果一定数量的batch后，验证集的损失没有下降，则提前结束训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        # 提前结束训练
        if flag:
            break
    # 模型训练结束后，使用matplotlib绘制训练过程中训练集和验证集的准确率和损失值变化趋势
    plt.figure()
    plt.plot(Iter_list, train_acc_list, label='Train Accuracy')
    plt.plot(Iter_list, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Iter_list, train_loss_list, label='Train Loss')
    plt.plot(Iter_list, val_loss_list, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.legend()
    plt.show()
    # 使用测试集对模型进行最终评估
    test(config, model, test_iter)


# 使用测试集对模型进行评估的函数
def test(config, model, test_iter):
    # 加载训练过程中保存的最佳模型参数
    model.load_state_dict(torch.load(config.save_path))
    # 将模型设置为评估模式
    model.eval()
    # 记录测试开始的时间
    start_time = time.time()
    # 在测试集上评估模型并获取准确率，损失值，混淆矩阵和分类报告
    test_acc, test_loss, test_confusion, test_report = evaluate(config, model, test_iter, test=True)
    # 输出测试损失和准确率信息
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    # 输出精确率，召回率和F1分数
    print("Precision, Recall and F1-Score...")
    print(test_report)

    # 输出混淆矩阵
    print("Confusion Matrix...")
    print(test_confusion)
    # 用pandas创建一个DataFrame来存储混淆矩阵的值，行列名为类别标签
    df = pd.DataFrame(test_confusion, index=config.class_list, columns=config.class_list)
    # 使用seaborn绘制混淆矩阵的热力图
    sns.heatmap(df, cmap="Blues", annot=True, fmt="d")
    # 设置热力图的标题以及x和y轴标签
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # 显示热力图
    plt.show()
    # 计算并打印测试所用时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # 将测试结果写入文件保存
    with open('./Datas/' + config.dataset + '/saved_dict/result.txt', 'a', encoding='utf-8') as f:
        f.write(str(test_acc) + '  ' + str(test_loss) + '\n')
    # 关闭文件
    f.close()


# 在给定的数据集上评估模型的性能
def evaluate(config, model, data_iter, test=False):
    # 将模型设置为评估模式
    model.eval()
    # 初始化总损失
    loss_total = 0
    # 初始化用于记录预测结果和实际标签的数组
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # 使用torch.no_grad()来停止计算和存储梯度，以节省内存和计算资源
    with torch.no_grad():
        # 迭代数据集
        for i, (texts, labels) in enumerate(data_iter):
            # 前向传播，获取模型输出
            outputs = model(texts)
            # 计算本批次的损失
            loss = F.cross_entropy(outputs, labels)
            # 累加损失
            loss_total += loss
            # 将标签从tensor转换为numpy数组
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # 更新预测结果和标签的总数组
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # 计算总准确率
    acc = metrics.accuracy_score(labels_all, predict_all)

    # 如果是测试模式，则额外计算并返回混淆矩阵和分类报告
    if test:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        return acc, loss_total / len(data_iter), confusion, report
    # 如果不是测试模式，则只返回准确率和平均损失
    return acc, loss_total / len(data_iter)
