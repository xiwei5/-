from model import *
test_path = glob.glob(r'.\mchar_test_a/*.png')
test_label = [[1]] * len(test_path)
test_path.sort()

def test_predict(net, data, abc, cuda, visualize, batch_size=50):
    """
    使用训练好的CRNN模型进行预测。

    参数：
    net (nn.Module): 训练好的CRNN模型。
    data (Dataset): 测试数据集。
    abc (str): 字符集字符串。
    cuda (bool): 是否使用CUDA设备。
    visualize (bool): 是否可视化预测结果（此参数未在函数中使用，可能为预留参数）。
    batch_size (int): 批量大小。

    返回：
    list: 预测结果列表。
    """
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=text_collate)

    count = 0  # 总样本数
    tp = 0  # 正确预测的样本数
    avg_ed = 0  # 平均编辑距离
    out = []  # 预测结果列表
    iterator = tqdm_notebook(data_loader)  # 使用tqdm_notebook显示进度条
    for sample in iterator:
        imgs = Variable(sample["img"])  # 获取图像数据
        if cuda:
            imgs = imgs.cuda()  # 如果使用CUDA，则将图像数据移动到CUDA设备
        out += net(imgs, decode=True)  # 前向传播并解码输出，将结果添加到预测结果列表
    return out  # 返回预测结果列表


# 加载CRNN模型
model = load_model('0123456789', seq_proj=[7, 30], backend='resnet18', snapshot='crnn_resnet18_0123456789_best',
                   cuda=True)

# 定义数据预处理变换
transform = Compose([
    Resize(size=(200, 100))  # 调整图像大小
])
# 加载测试数据集
test_data = TextDataset(test_path, test_label, transform=transform)

# 将模型设置为评估模式
model.training = False
test_predict=test_predict(model, test_data, '0123456789', True, False, batch_size=50)

# 读取提交文件模板
df_submit = pd.read_csv(r".\mchar_sample_submit_A.csv")
# 将预测结果添加到提交文件中
df_submit['file_code'] = test_predict
# 保存更新后的提交文件
df_submit.to_csv(r".\mchar_sample_submit_A.csv", index=None)


# 可视化预测结果
def visualize_predictions(test_path, predictions, num_samples=5):
    # 随机选择几张图片
    indices = random.sample(range(len(test_path)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i, idx in enumerate(indices):
        img_path = test_path[idx]
        img = plt.imread(img_path)
        pred = predictions[idx]

        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(f'Predicted: {pred}')
        axes[i].axis('off')

    plt.show()


# 调用可视化函数
visualize_predictions(test_path, test_predict, num_samples=5)