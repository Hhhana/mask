###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition
#from tensorflow.reverse as tf.reverse

# 1.加载数据并进行数据处理
# 数据集路径
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/image"
def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):  # 此处参数值是默认值
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True) # 加载训练数据
    valid_data_loader = DataLoader(test_dataset, batch_size=16,shuffle=True) # 加载验证/测试数据

    return train_data_loader, valid_data_loader


# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载
torch.set_num_threads(1) # 设置线程数

# 加载 MobileNet 的预训练模型权重
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # 选择运算设备
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=64)
modify_x, modify_y = torch.ones((64, 3, 160, 160)), torch.ones((64))

epochs = 30
model = MobileNetV1(classes=2).to(device) # 二分类问题
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 优化器：lr即学习率0.0001
# optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
print('加载完成...')

# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'max', 
                                                 factor=0.5,
                                                 patience=2)


# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹
# 损失函数：交叉熵函数
criterion = nn.CrossEntropyLoss()  
best_loss = 1e9
best_val_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []  # 存储损失函数值
acc_list = []  # 存储验证集上accuracy值

for epoch in range(epochs):
    # 训练一次网络
    model.train()
    # 批处理
    print("epoch_idx:",epoch)
    for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
        print("batch_idx:",batch_idx)
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)

        # print(pred_y.shape)
        # print(y.shape)

        loss = criterion(pred_y, y) # 计算loss
        optimizer.zero_grad() # zero the gradient buffer梯度清零，以免影响其他batch
        loss.backward() # 后向传播，计算梯度
        optimizer.step() # 梯度更新

        if loss < best_loss: # 选择最小损失
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss
            
        #loss_list.append(loss) # 存储损失函数
        print('{{"metric": "loss", "value": {}}}'.format(loss))

    # 迭代一轮计算accuracy
    # 模型路径
    model_path = './results/temp.pth'
    # 临时保存net
    torch.save(model.state_dict(), model_path)
    # 加载网络
    temp_net = Recognition(model_path)
    correct = 0
    total = 0
    best_val_loss = 1e9
    with torch.no_grad():# 执行的固定操作
        for index, (x, labels) in enumerate(valid_data_loader):
            print("index:",index)
            temp_correct = 0
            size = labels.size(0)
            print("size:",size)
            total += size

            # 可视化valid loss
            x = x.to(device)
            y = labels.to(device)
            pred_y = model(x)
            val_loss = criterion(pred_y, y) # 计算val_loss
            
            if val_loss < best_val_loss: # 选择最小损失
                best_model_weights = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss

            print('{{"metric": "valid loss", "value": {}}}'.format(val_loss))

            for i in range(size):
                #print("x[i]尺寸:", x[i].shape)
                #x[i] = x[i].reshape(160,160,3) # 交换维度3*160*160->160*160*3
                #print("图片尺寸2:", x[i].shape)
                #a = np.array(x[i]) # 转换为ndarray
                #print("a尺寸:", a.shape)
                #x[i] = T.ToTensor()(a).unsqueeze(0) # 在第一维增加一维，构成4维tensor，参数必须为PIL或者ndarray
                a = x[i].unsqueeze(0) # 在第一维增加一维，构成4维tensor，参数必须为PIL或者ndarray
                #print("图片尺寸:", x[i].unsqueeze(0).shape)
                temp_net.mobilenet.eval()
                predict_label = temp_net.mobilenet(a).cpu().data.numpy()
                current_class = temp_net.classes[np.argmax(predict_label).item()]
                temp_correct += (current_class == "mask")
            correct += temp_correct
            print("correct:", temp_correct)
            print("total:", total)
        print("总的correct:", correct)
        print("总的total:", total)
        print('{{"metric": "accuracy", "value": {}}}'.format(correct / total))
        #acc_list.append(correct / total)

    #print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || accuracy: %.4f' % (correct / total))

# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。
torch.save(best_model_weights, './results/temp40.pth') #model.state_dict()
print('Finish Training.') 


