import json
import matplotlib.pyplot as plt
import random
import pandas as pd
import torchvision.models as models
from collections import OrderedDict
from torch.utils.data import Dataset
import cv2
import editdistance
import os
import numpy as np
from tqdm import tqdm, tqdm_notebook
from torchvision.transforms import Compose
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import glob

def text_collate(batch):
    """
    自定义批处理函数，用于将样本列表转换为批处理张量。

    参数：
    batch (list): 包含多个样本的列表，每个样本是一个字典。

    返回：
    dict: 包含批处理张量的字典。
    """
    img = list()
    seq = list()
    seq_len = list()
    for sample in batch:
        # 将图像转换为张量并添加到列表中
        img.append(torch.from_numpy(sample["img"].transpose((2, 0, 1))).float())
        # 将序列数据添加到列表中
        seq.extend(sample["seq"])
        # 将序列长度数据添加到列表中
        seq_len.append(sample["seq_len"])
    # 将图像列表堆叠成一个张量
    img = torch.stack(img)
    # 将序列数据和序列长度数据转换为张量
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()
    # 返回包含批处理张量的字典
    batch = {"img": img, "seq": seq, "seq_len": seq_len}
    return batch


class ToTensor(object):
    """
    将样本中的图像和序列数据转换为张量。
    """

    def __call__(self, sample):
        """
        调用函数，执行转换操作。

        参数：
        sample (dict): 包含图像和序列数据的样本字典。

        返回：
        dict: 包含转换后的张量数据的字典。
        """
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class Resize(object):
    """
    调整样本中图像的大小。

    参数：
    size (tuple): 目标图像大小，格式为 (宽度, 高度)。
    """

    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, sample):
        """
        调用函数，执行调整大小操作。

        参数：
        sample (dict): 包含图像数据的样本字典。

        返回：
        dict: 包含调整大小后的图像数据的字典。
        """
        sample["img"] = cv2.resize(sample["img"], self.size)
        return sample


class Rotation(object):
    """
    随机旋转样本中的图像。

    参数：
    angle (float): 旋转角度范围，单位为度。
    fill_value (int): 旋转后空白区域的填充值。
    p (float): 执行旋转操作的概率。
    """

    def __init__(self, angle=5, fill_value=0, p=0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        """
        调用函数，执行随机旋转操作。

        参数：
        sample (dict): 包含图像数据的样本字典。

        返回：
        dict: 包含旋转后的图像数据的字典。
        """
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        ang_rot = np.random.uniform(self.angle) - self.angle / 2
        transform = cv2.getRotationMatrix2D((w / 2, h / 2), ang_rot, 1)
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w, h), borderValue=self.fill_value)
        return sample


class Translation(object):
    """
    随机平移样本中的图像。

    参数：
    fill_value (int): 平移后空白区域的填充值。
    p (float): 执行平移操作的概率。
    """

    def __init__(self, fill_value=0, p=0.5):
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        """
        调用函数，执行随机平移操作。

        参数：
        sample (dict): 包含图像数据的样本字典。

        返回：
        dict: 包含平移后的图像数据的字典。
        """
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        trans_range = [w / 10, h / 10]
        tr_x = trans_range[0] * np.random.uniform() - trans_range[0] / 2
        tr_y = trans_range[1] * np.random.uniform() - trans_range[1] / 2
        transform = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w, h), borderValue=self.fill_value)
        return sample


class Scale(object):
    """
    随机缩放样本中的图像。

    参数：
    scale (list): 缩放比例范围，格式为 [最小比例, 最大比例]。
    fill_value (int): 缩放后空白区域的填充值。
    p (float): 执行缩放操作的概率。
    """

    def __init__(self, scale=[0.5, 1.2], fill_value=0, p=0.5):
        self.scale = scale
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        """
        调用函数，执行随机缩放操作。

        参数：
        sample (dict): 包含图像数据的样本字典。

        返回：
        dict: 包含缩放后的图像数据的字典。
        """
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        scale = np.random.uniform(self.scale[0], self.scale[1])
        transform = np.float32([[scale, 0, 0], [0, scale, 0]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w, h), borderValue=self.fill_value)
        return sample




class TextDataset(Dataset):
    """
    文本数据集类，用于加载和处理图像及其对应的标签。

    参数：
    data_path (list): 包含图像文件路径的列表。
    data_label (list): 包含图像对应文本标签的列表。
    transform (callable, optional): 可选的数据转换函数。
    """

    def __init__(self, data_path, data_label, transform=None):
        super().__init__()
        self.data_path = data_path  # 图像文件路径列表
        self.data_label = data_label  # 图像对应的文本标签列表
        self.transform = transform  # 数据转换函数

    def abc_len(self):
        """
        获取字符集的长度。

        返回：
        int: 字符集的长度。
        """
        return len('0123456789')  # 字符集为数字0-9

    def get_abc(self):
        """
        获取字符集。

        返回：
        str: 字符集字符串。
        """
        return '0123456789'  # 字符集为数字0-9

    def set_mode(self, mode):
        """
        设置数据集模式。

        参数：
        mode (str): 数据集模式（如训练模式、验证模式等）。
        """
        self.mode = mode  # 设置数据集模式

    def __len__(self):
        """
        获取数据集的长度。

        返回：
        int: 数据集的长度。
        """
        return len(self.data_path)  # 数据集长度为图像文件路径列表的长度

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。

        参数：
        idx (int): 样本的索引。

        返回：
        dict: 包含图像、序列、序列长度和增强标志的样本字典。
        """
        text = self.data_label[idx]  # 获取文本标签

        img = cv2.imread(self.data_path[idx])  # 读取图像
        seq = self.text_to_seq(text)  # 将文本转换为序列
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": 1}  # 创建样本字典
        if self.transform:
            sample = self.transform(sample)  # 如果有转换函数，则对样本进行转换
        return sample

    def text_to_seq(self, text):
        """
        将文本字符串转换为数字序列。

        参数：
        text (str): 文本字符串。

        返回：
        list: 数字序列列表。
        """
        seq = []
        for c in text:
            seq.append(self.get_abc().find(str(c)) + 1)  # 将字符转换为对应的索引+1
        return seq  # 返回数字序列列表


class CRNN(nn.Module):
    """
    卷积循环神经网络（CRNN）类，用于文本识别任务。

    参数：
    abc (str): 字符集字符串。
    backend (str): 预训练模型的后端，用于特征提取。
    rnn_hidden_size (int): RNN隐藏层的大小。
    rnn_num_layers (int): RNN层数。
    rnn_dropout (float): RNN层之间的dropout率。
    seq_proj (list): 序列投影的输入和输出维度。
    """

    def __init__(self,
                 abc='0123456789',
                 backend='resnet18',
                 rnn_hidden_size=64,
                 rnn_num_layers=1,
                 rnn_dropout=0,
                 seq_proj=[0, 0]):
        super(CRNN, self).__init__()

        self.abc = abc  # 字符集
        self.num_classes = len(self.abc)  # 类别数（字符集大小）

        # 加载预训练模型作为特征提取器
        self.feature_extractor = getattr(models, backend)(pretrained=True)
        # 构建CNN层
        self.cnn = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            self.feature_extractor.maxpool,
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            self.feature_extractor.layer3,
            self.feature_extractor.layer4
        )

        self.fully_conv = seq_proj[0] == 0  # 是否全卷积
        if not self.fully_conv:
            # 如果不是全卷积，则添加一个投影层
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)

        self.rnn_hidden_size = rnn_hidden_size  # RNN隐藏层大小
        self.rnn_num_layers = rnn_num_layers  # RNN层数
        # 构建RNN层
        self.rnn = nn.GRU(self.get_block_size(self.cnn),
                          rnn_hidden_size, rnn_num_layers,
                          batch_first=False,
                          dropout=rnn_dropout, bidirectional=True)
        # 构建线性层
        self.linear = nn.Linear(rnn_hidden_size * 2, self.num_classes + 1)
        self.softmax = nn.Softmax(dim=2)  # Softmax层

    def forward(self, x, decode=False):
        """
        前向传播函数。

        参数：
        x (Tensor): 输入张量。
        decode (bool): 是否解码输出。

        返回：
        Tensor: 输出张量。
        """
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        features = self.features_to_sequence(features)
        seq, hidden = self.rnn(features, hidden)
        seq = self.linear(seq)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                seq = self.decode(seq)
        return seq

    def init_hidden(self, batch_size, gpu=False):
        """
        初始化RNN隐藏状态。

        参数：
        batch_size (int): 批量大小。
        gpu (bool): 是否使用GPU。

        返回：
        Variable: 隐藏状态变量。
        """
        h0 = Variable(torch.zeros(self.rnn_num_layers * 2,
                                  batch_size,
                                  self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        """
        将特征图转换为序列。

        参数：
        features (Tensor): 特征张量。

        返回：
        Tensor: 序列张量。
        """
        features = features.mean(2)
        b, c, w = features.size()
        features = features.reshape(b, c, 1, w)
        b, c, h, w = features.size()
        assert h == 1, "the height of out must be 1"
        if not self.fully_conv:
            features = features.permute(0, 3, 2, 1)
            features = self.proj(features)
            features = features.permute(1, 0, 2, 3)
        else:
            features = features.permute(3, 0, 2, 1)
        features = features.squeeze(2)
        return features

    def get_block_size(self, layer):
        """
        获取块大小。

        参数：
        layer (nn.Module): 网络层。

        返回：
        int: 块大小。
        """
        return layer[-1][-1].bn2.weight.size()[0]

    def pred_to_string(self, pred):
        """
        将预测结果转换为字符串。

        参数：
        pred (numpy.ndarray): 预测结果。

        返回：
        str: 解码后的字符串。
        """
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        out = ''.join(self.abc[i] for i in out)
        return out

    def decode(self, pred):
        """
        解码预测结果。

        参数：
        pred (Tensor): 预测张量。

        返回：
        list: 解码后的字符串列表。
        """
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq


def load_weights(target, source_state):
    """
    加载预训练模型的权重到目标模型。

    参数：
    target (nn.Module): 目标模型。
    source_state (dict): 预训练模型的权重字典。
    """
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        # 如果目标模型的键在源权重字典中存在，并且权重大小相同，则加载权重
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            # 否则，使用目标模型的原始权重
            new_dict[k] = v
    # 加载新的权重字典到目标模型
    target.load_state_dict(new_dict)


def load_model(abc, seq_proj=[0, 0], backend='resnet18', snapshot=None, cuda=True):
    """
    加载CRNN模型，并可选择加载预训练权重和移动到CUDA设备。

    参数：
    abc (str): 字符集字符串。
    seq_proj (list): 序列投影的输入和输出维度。
    backend (str): 预训练模型的后端。
    snapshot (str): 预训练模型的文件路径。
    cuda (bool): 是否使用CUDA设备。

    返回：
    nn.Module: 加载后的CRNN模型。
    """
    net = CRNN(abc=abc, seq_proj=seq_proj, backend=backend)
    # 如果提供了预训练模型的文件路径，则加载权重
    if snapshot is not None:
        load_weights(net, torch.load(snapshot))
    # 如果使用CUDA设备，则将模型移动到CUDA设备
    if cuda:
        net = net.cuda()
    return net


class StepLR(object):
    """
    自定义学习率衰减策略，每经过一定步数后衰减学习率。

    参数：
    optimizer (torch.optim.Optimizer): 优化器。
    step_size (int): 学习率衰减的步数间隔。
    max_iter (int): 最大迭代次数。
    """

    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        # 获取每个参数组的初始学习率
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        """
        获取当前的学习率。

        返回：
        float: 当前的学习率。
        """
        return self.optimizer.param_groups[0]['lr']

    def step(self, last_iter=None):
        """
        更新学习率。

        参数：
        last_iter (int, optional): 上一次迭代的步数。
        """
        if last_iter is not None:
            self.last_iter = last_iter
        # 如果达到最大迭代次数，则重置迭代步数
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        # 更新迭代步数
        self.last_iter = (self.last_iter + 1) % self.max_iter
        # 对每个参数组更新学习率
        for ids, param_group in enumerate(self.optimizer.param_groups):
            # 计算新的学习率并更新
            param_group['lr'] = self.base_lrs[ids] * 0.8 ** (self.last_iter // self.step_size)


def test(net, data, abc, cuda, batch_size=50):
    """
    测试CRNN模型的准确率和平均编辑距离。

    参数：
    net (nn.Module): CRNN模型。
    data (Dataset): 测试数据集。
    abc (str): 字符集字符串。
    cuda (bool): 是否使用CUDA设备。
    batch_size (int): 批量大小。

    返回：
    tuple: 包含准确率和平均编辑距离的元组。
    """
    # 创建DataLoader来批量处理数据
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

    count = 0  # 总样本数
    tp = 0  # 正确预测的样本数
    avg_ed = 0  # 平均编辑距离
    # 使用tqdm_notebook显示进度条
    iterator = tqdm_notebook(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])  # 获取图像数据
        if cuda:
            imgs = imgs.cuda()  # 如果使用CUDA，则将图像数据移动到CUDA设备
        out = net(imgs, decode=True)  # 前向传播并解码输出
        gt = (sample["seq"].numpy() - 1).tolist()  # 获取真实的序列数据并转换为列表
        lens = sample["seq_len"].numpy().tolist()  # 获取序列长度数据并转换为列表
        pos = 0  # 序列位置指针
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos + lens[i]])  # 根据序列长度获取真实的文本标签
            pos += lens[i]  # 更新序列位置指针
            if gts == out[i]:
                tp += 1  # 如果预测文本与真实文本相同，则增加正确预测的样本数
            else:
                avg_ed += editdistance.eval(out[i], gts)  # 否则，计算编辑距离并累加到平均编辑距离
            count += 1  # 增加总样本数
        # 更新进度条的描述信息，显示当前的准确率和平均编辑距离
        iterator.set_description("acc: {0:.4f}; avg_ed: {1:.4f}".format(tp / count, avg_ed / count))

    # 计算最终的准确率和平均编辑距离
    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed  # 返回准确率和平均编辑距离
train_json = json.load(open(r".\mchar_train.json"))
train_label = [train_json[x]['label'] for x in train_json.keys()]
train_path = [r'.\mchar_train/' + x for x in train_json.keys()]

val_json = json.load(open(r".\mchar_val.json"))
val_label = [val_json[x]['label'] for x in val_json.keys()]
val_path = [r'.\mchar_val/' + x for x in val_json.keys()]


def test_main(
        abc='0123456789',
        seq_proj="7x30",
        backend="resnet18",
        snapshot=None,
        input_size="200x100",
        base_lr=1e-3,
        step_size=1000,
        max_iter=10000,
        batch_size=20,
        output_dir='./',
        test_epoch=1,
        test_init=None,
        gpu='0'):
    """
    主测试函数，用于训练和测试CRNN模型。

    参数：
    abc (str): 字符集字符串。
    seq_proj (str): 序列投影的维度，格式为"高度x宽度"。
    backend (str): 预训练模型的后端。
    snapshot (str): 预训练模型的文件路径。
    input_size (str): 输入图像的大小，格式为"宽度x高度"。
    base_lr (float): 初始学习率。
    step_size (int): 学习率衰减的步数间隔。
    max_iter (int): 最大迭代次数。
    batch_size (int): 批量大小。
    output_dir (str): 输出目录。
    test_epoch (int): 每隔多少个epoch进行一次测试。
    test_init (bool): 是否在初始化后进行测试。
    gpu (str): 使用的GPU设备编号。
    """
    # 设置可见的CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # 根据GPU设备编号判断是否使用CUDA
    cuda = True if gpu is not '' else False

    # 解析输入图像的大小
    input_size = [int(x) for x in input_size.split('x')]
    # 定义数据预处理变换
    transform = Compose([
        Rotation(),
        Translation(),
        # Scale(),
        Resize(size=(input_size[0], input_size[1]))
    ])

    # 加载训练和验证数据集
    data = TextDataset(train_path, train_label, transform=transform)
    data_val = TextDataset(val_path, val_label, transform=transform)

    # 解析序列投影的维度
    seq_proj = [int(x) for x in seq_proj.split('x')]
    # 加载CRNN模型
    net = load_model(abc, seq_proj, backend, snapshot, cuda)
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    # 定义学习率调度器
    lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)
    # 定义损失函数
    loss_function = CTCLoss(zero_infinity=True)

    # 初始化最佳准确率和epoch计数
    acc_best = 0
    epoch_count = 0
    # 开始训练循环
    while (epoch_count < 20):
        # 如果达到测试条件，则进行测试
        if (test_epoch is not None and epoch_count != 0 and epoch_count % test_epoch == 0) or (
                test_init and epoch_count == 0):
            print("Test phase")
            # 设置数据集为测试模式
            data.set_mode("test")
            # 将模型设置为评估模式
            net = net.eval()
            # 进行测试并获取准确率和平均编辑距离
            acc, avg_ed = test(net, data_val, data.get_abc(), cuda, 50)
            # 将模型恢复为训练模式
            net = net.train()
            # 恢复数据集为训练模式
            data.set_mode("train")
            # 如果当前准确率优于最佳准确率，则保存模型
            if acc > acc_best:
                if output_dir is not None:
                    torch.save(net.state_dict(),
                               os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_best"))
                acc_best = acc
            # 打印当前的准确率、最佳准确率和平均编辑距离
            print("acc: {}\tacc_best: {}; avg_ed: {}".format(acc, acc_best, avg_ed))

        # 创建DataLoader来批量处理数据
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=text_collate)
        loss_mean = []  # 初始化损失列表
        iterator = tqdm(data_loader)  # 使用tqdm显示进度条
        iter_count = 0  # 初始化迭代计数
        for sample in iterator:
            # 为了支持多GPU，跳过不能均匀分配的批次
            if sample["img"].size(0) % len(gpu.split(',')) != 0:
                continue
            optimizer.zero_grad()  # 清除优化器的梯度
            imgs = Variable(sample["img"])  # 获取图像数据
            labels = Variable(sample["seq"]).view(-1)  # 获取标签数据并调整形状
            label_lens = Variable(sample["seq_len"].int())  # 获取标签长度数据
            if cuda:
                imgs = imgs.cuda()  # 如果使用CUDA，则将图像数据移动到CUDA设备
            preds = net(imgs).cpu()  # 前向传播并获取预测结果
            pred_lens = Variable(torch.Tensor([preds.size(0)] * batch_size).int())  # 获取预测长度数据

            # 计算损失
            loss = loss_function(preds, labels, pred_lens, label_lens)
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm(net.parameters(), 10.0)  # 梯度裁剪
            loss_mean.append(loss.item())  # 记录损失值
            status = "epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count, lr_scheduler.last_iter,
                                                                                   lr_scheduler.get_lr(),
                                                                                   np.mean(loss_mean), loss.item())
            iterator.set_description(status)  # 更新进度条的描述信息
            optimizer.step()  # 更新优化器的参数
            lr_scheduler.step()  # 更新学习率调度器
            iter_count += 1  # 增加迭代计数
        # 保存模型的最新状态
        if output_dir is not None:
            torch.save(net.state_dict(),
                       os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_last"))
        epoch_count += 1  # 增加epoch计数

    return 1  # 返回状态码







