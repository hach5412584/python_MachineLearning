import os
import math
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from AOI_train import CreateList, CustomDataset, VGG ,ResNet ,DenseNet
from torchvision.models import VGG16_BN_Weights

# Paths
dir_img_test = 'C:/dataset/aoi/test_images/test_images'
path_label_test = 'C:/dataset/aoi/test.csv'
path_model = 'C:/train/AOI_train/DenseNet_best_model_23.pth'
save_submit = './submit/{}_submit.csv'.format(path_model.split('/')[-1].replace('.pth', ''))

# Parameters
cuda = True
workers = 2
batch_size = 128

def main():
    # Load the Model
    net = DenseNet(num_classes=10)
    #net = VGG('aoi', weights=VGG16_BN_Weights.DEFAULT)
    save = torch.load(path_model)

    # 检查并打印保存字典的键
    print("Loaded keys from saved model:", save.keys())

    # 加载模型状态字典
    net.load_state_dict(save)
    net.eval()

    # Send model into gpu memory
    if cuda:
        net.cuda()

    # Prepare the data
    test_list = CreateList(dir_img_test, path_label_test, shuffle=False, train=False)

    transform = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    fake_list = [i for i in range(len(test_list.img))]

    test_dataset = CustomDataset(test_list.img,
                                 label_list=fake_list,
                                 transform=transform['test'])

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True)

    # Predict test images
    # Collect prediction values
    test_predict = []
    net.eval()
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            if cuda:
                images = images.cuda()

            out = net(images)  # forward
            _, pred = torch.max(out.data, 1)
            test_predict += pred.cpu().numpy().tolist()

    # Check number of class of predictions
    print(f"Number of unique classes in predictions: {len(set(test_predict))}")
    # Check whether the number of predictions match test images
    print(f"Number of predictions matches number of test images: {len(test_predict) == len(test_list.filename)}")

    # Create submit data
    df_submit = pd.DataFrame({'ID': test_list.filename,
                              'Label': test_predict})

    df_submit.to_csv(save_submit,
                     header=True,
                     sep=',',
                     encoding='utf-8',
                     index=False)

if __name__ == '__main__':
    main()
