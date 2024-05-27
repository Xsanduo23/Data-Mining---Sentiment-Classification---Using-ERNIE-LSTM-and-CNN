# coding: UTF-8
import random
import jieba
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
#
PUNCTUATIONS = ['"', '!', '（', '—', ':']

NUM_AUGS = [1, 2, 4, 8]
PUNC_RATIO = 0.2


# 添加 干扰词
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)

    return new_line


# 将英文全都小写话，没用到现在
def chinese_en_tokenzizer(content):
    content = list(jieba.cut(content, cut_all=False))
    new_content = []
    for words in content:
        count = 0
        for word in words:
            if '\u4e00' <= word <= '\u9fa5':
                count += 1
                new_content.append(word)

        if count == 0:
            new_content.append(words.lower())

    # new_content = insert_punctuation_marks(new_content)
    new_content = ''.join(new_content)
    return new_content


# 根据给定的配置构建数据集
def build_dataset(config):
    # 用于记录句子长度信息
    lengths = []
    # 是否使用停顿词
    if config.use_stop_words:
        print("停顿词：")
        stop_words = [x.strip() for x in open('./Datas/stopword/stopword.txt', encoding='gbk').readlines()]
        print(stop_words)

    # 用于从文件中加载数据集
    def load_dataset(path, pad_size=32, istrain=False):
        # 从文件中加载类别列表
        class_list = [x.strip() for x in
                      open('./Datas/' + config.dataset + '/data/class.txt', encoding='utf-8').readlines()]
        # 用于装载处理好的数据
        contents = []
        # 打开指定的数据集文件
        with open(path, 'r', encoding='UTF-8') as f:
            # 逐行读取数据
            for index, line in tqdm(enumerate(f)):
                lin = line.strip()
                # 如果行为空，则忽略
                if not lin:
                    continue
                try:
                    # 尝试根据特定分隔符分割内容和标签
                    content, label = lin.split(' __!__ ')
                except:
                    # 出错则打印有误的行
                    print(lin)
                if istrain:
                    # 进行中英文大小写处理(默认不用)
                    content = chinese_en_tokenzizer(content)

                try:
                    # 获取标签的索引
                    label = class_list.index(label)
                except:
                    # 出错则打印有误的行
                    print(lin)
                if config.use_stop_words:
                    # 停顿词处理
                    words = list(jieba.lcut(content))
                    words = [word for word in words if word not in stop_words]
                    content = ' '.join(words)
                # 使用配置中的分词器进行分词
                token = config.tokenizer.tokenize(content)
                # 将CLS标记添加到分词结果的开头
                token = [CLS] + token
                # 记录分词后句子的真实长度
                seq_len = len(token)
                # 记录长度信息供后续分析
                lengths.append(seq_len)
                # 初始的掩码，后续将会根据句子长度进行调整
                mask = []
                # 将分词结果转换为对应的ID
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                # 如果设置了填充的最大长度
                if pad_size:
                    if len(token) < pad_size:
                        # 如果句子长度小于设置的最大长度则进行填充
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        # 如果句子长度大于等于设置的最大长度则进行截断
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                # 将处理好的一条数据添加到contents中
                contents.append((token_ids, int(label), seq_len, mask))
        # 最终返回处理完成的数据集
        return contents

    # 加载训练、验证和测试数据
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    # 对句子长度小于设置的填充长度的进行统计
    new = []
    for i in lengths:
        if i < config.pad_size:
            new.append(i)
    # 打印统计结果
    print('小于{}百分比：{}'.format(config.pad_size, len(new) / len(lengths)))
    # 返回构建好的训练集、验证集、测试集
    return train, dev, test


# 加载对应的数据集（有预测未知数据时使用，模型训练用不到）
def build_String(config, str):
    lengths = []

    def load_dataset(str, pad_size=config.pad_size, istrain=False):

        contents = []
        for content in str:
            try:
                content = content.lower()
                token = config.tokenizer.tokenize(content)
            except:
                token = []

            token = [CLS] + token

            seq_len = len(token)
            lengths.append(seq_len)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, 0, seq_len, mask))
        return contents

    return load_dataset(str)


# 定义一个数据迭代器类，用于方便地批量处理和迭代数据集
class DatasetIterater(object):
    # 初始化迭代器对象
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size  # 批次大小
        self.batches = batches  # 数据批次
        self.n_batches = len(batches) // batch_size  # 计算总共有多少完整批次
        self.residue = False  # 记录是否有不完整的批次，即数据集大小不是批次大小的整数倍
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 用于记录当前迭代到哪个批次
        self.device = device  # 指定要将数据移动到的设备（比如CPU或GPU）

    # 将数据批次转换成Tensor，并移动到指定的设备
    def _to_tensor(self, datas):
        # 将输入x转换为张量并移至设备
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # 将标签y转换为张量并移至设备
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # 将序列长度seq_len转换为张量并移至设备（超过pad_size的设为pad_size）
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # 将掩码mask转换为张量并移至设备
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    # 获取下一个批次的数据
    def __next__(self):
        # 如果有不完整的批次，并且当前索引已到最后一个完整批次
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            # 将数据转换为张量
            batches = self._to_tensor(batches)
            return batches
        # 如果索引已经超过了批次数，重置索引并停止迭代
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        # 否则，获取下一个完整批次的数据
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            # 将数据转换为张量
            batches = self._to_tensor(batches)
            return batches

    # 返回迭代器自身
    def __iter__(self):
        return self

    # 返回批次总数，如果有不完整的批次则加1
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
