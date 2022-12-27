from get_model import resnext50
from Metric_Learning_DataLoader import CustomDataset

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numba

'''
    현재 pip list에서 torch 관련 패키지를 다음과 같이 맞추어야 할 필요가 있음.
    torch       1.11.0  ->  torch       1.11.0+cu117
    torchaudio  0.11.0  ->  torchaudio  0.11.0+cu117
    torchvision 0.12.0  ->  torchvision 0.12.0+cu117
'''


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss\\resnet50_2022-10-13_23-1-12_50.pt'


model = resnext50()
model.load_state_dict(torch.load(PATH))
model = model.to(device)
model.eval()

path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
])

batch_size = 32
test_data = CustomDataset(path, train_transform=transform)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

result = []
dist_result = []
labels_result = []

print(f"Evaluation Start...")
for data in test_dataloader:
    imgs1, imgs1_with_labels, imgs2, imgs2_with_labels, labels = data
    imgs1, imgs2, labels = imgs1.to(device), imgs2.to(device), labels.to(device)

    outputs1 = model(imgs1)
    outputs2 = model(imgs2)
    distances = F.pairwise_distance(outputs1, outputs2, keepdim=True)

    distances = distances.detach().cpu().numpy() * 1e2
    labels = np.expand_dims(labels.detach().cpu().numpy(), axis=-1)

    # 거리 정보, 라벨 정보 따로 가짐
    dist_result = np.append(dist_result, distances)
    labels_result = np.append(labels_result, labels)


# Result를 다 얻었으면, 그림을 그려보기
print(max(dist_result), min(dist_result))
print("make histogram...")

bins = np.arange(np.min(dist_result)-1., np.max(dist_result)+1., 1)

same_list = []
different_list = []
for idx in range(len(labels_result)):
    if labels_result[idx] == 0:
        same_list = np.append(same_list, dist_result[idx])
    else:
        different_list = np.append(different_list, dist_result[idx])

same_hist, bins = np.histogram(same_list, bins)
diff_hist, bins = np.histogram(different_list, bins)

####################################################################################
# label = min(data), label = max(data)로 나누기
max_distance = int(np.max(dist_result))
max_threshold = 0
max_F1_score = 0
for temp_threshold in range(max_distance):
    TP, TN, FP, FN = 0, 0, 0, 0
    for dist in same_list:
        if dist <= temp_threshold:
            TP += 1
        else:
            FN += 1
    for dist in different_list:
        if dist >= temp_threshold:
            TN += 1
        else:
            FP += 1

    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    F1_score = (2*TP)/(2*TP + FP + FN)

    print(f"f1 score: {F1_score} / temp_threshold: {temp_threshold}")
    if max_F1_score < F1_score:
        max_F1_score = F1_score
        max_threshold = temp_threshold
print(f"The best threshold: {max_threshold}")
####################################################################################


'''
    여기에서 최적의 threshold값 구하는 거 진행해보기
    일단은 서로 겹치는 부분이 없으므로 threshold = 125
    Neal에게 배우기
'''



# print("Draw histogram...")
plt.hist(same_list, bins, rwidth = 0.5, color = 'green', alpha = 0.5, label = 'Same')
plt.hist(different_list, bins, rwidth = 0.5, color = 'red', alpha = 0.5, label = 'Different')
# # plt.grid()
plt.xlabel('Distance', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14)
plt.show()
