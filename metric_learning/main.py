from get_model import resnext50
from Contrastive_loss import ContrastiveLoss, ArcFace
from Metric_Learning_DataLoader import CustomDataset
from torch.utils.data import DataLoader
from optimizers import get_optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt

class ModelManager(object):

    def __init__(self,
                 save_path:str,
                 load_path:str,
                 model:nn.Module,
                 is_weight_loaded = True,
                 criterion1 = ContrastiveLoss(),
                 criterion2 = nn.CrossEntropyLoss(),
                 classifier = ArcFace,
                 num_of_class = 1024,
                 optimizer = "adam",
                 learning_rate = 0.002,
                 epochs=3):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.load_path = load_path
        self.model = model.to(self.device)
        self.classifier = classifier(in_features=model.fc.out_features, out_features=num_of_class).to(self.device)
        self.best_threshold = None

        if is_weight_loaded:
            print(f"loading the weight to model...")
            print(f"model path:{self.load_path}")
            self.load_model()
            print(f"Successfully weight is loaded")

        # model.load_state_dict(torch.load(load_path))
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        '''
            get_optimizer 만들기
        '''
        self.optimizer = get_optimizer(optimizer, self.model, learning_rate)

        model_params = [params for params in self.model.parameters()]
        classifier_params = [params for params in self.classifier.parameters()]
        torch.optim.Adam(model_params + classifier_params, lr=0.002)
        self.epochs = epochs

    def get_dataset(self,
                    path,
                    batch_size = 32,
                    num_workers = 4,
                    transform = transforms.Compose([
                                    transforms.Resize((150,150)),
                                    transforms.ToTensor(),
                    ])):

        self.dataset_path = path
        self.transform = transform
        self.batch_size = batch_size
        self.train_data = CustomDataset(path, train_transform=self.transform)
        self.test_data = CustomDataset(path, train_transform=self.transform)


        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


    def train(self):
        print(f"\nmodel learning start...(device state is {self.device})")

        self.model.train()
        self.save_epoch = 0
        for epoch in range(1, self.epochs+1):
            self.save_epoch = epoch
            running_loss = 0.0

            for i, data in enumerate(tqdm(self.train_dataloader)):
                imgs1, imgs1_with_labels, imgs2, imgs2_with_labels, labels = data

                imgs1, imgs2, labels = imgs1.to(self.device), imgs2.to(self.device), labels.to(self.device)
                imgs1_with_labels = imgs1_with_labels.to(self.device)
                imgs2_with_labels = imgs2_with_labels.to(self.device)

                self.optimizer.zero_grad()

                outputs1 = self.model(imgs1)
                outputs2 = self.model(imgs2)

                loss1 = self.criterion1(outputs1, outputs2, labels)
                loss2 = (self.criterion2(outputs1, imgs1_with_labels) + self.criterion2(outputs2, imgs2_with_labels))
                # loss2_left = (self.criterion2(self.classifier(outputs1, imgs1_with_labels), imgs1_with_labels))
                # loss2_right = (self.criterion2(self.classifier(outputs1, imgs2_with_labels), imgs2_with_labels))
                loss = 0.5 * (loss1 + loss2) # 추후에 loss의 비율을 다르게 두는 hyperparameter 도입해보기

                '''
                    running_loss는 72.xxx 부근에서 떨어지지 않으나,
                    실질적인 same, diffenent는 파악을 잘해낸다.
                '''
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()


            self.save_model()
            print(f"\n[{self.epochs}/{epoch}] -> running loss:{running_loss}")
            print(f"\nmodel is saved successfully.")

        print(f"model learning is finished")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_path))


    def save_model(self):
        x = datetime.datetime.now()
        hour, minute, second = x.hour, x.minute, x.second

        if hour < 10:
            hour = '0' + str(x.hour)
        if minute < 10:
            minute = '0' + str(x.minute)
        if second < 10:
            second = '0' + str(x.second)

        torch.save(self.model.state_dict(), f'{self.save_path}\\resnet50_{x.year}-{x.month}-{x.day}_{hour}-{minute}-{second}_{self.save_epoch}.pt')
        print(f"\nmodel is saved successfully.")



    def evaluate(self, show_plot = True):
        self.model.eval()
        ####################################
        # TODO: 여기서 최적의 Threshold 구하기
        ####################################
        test_data = CustomDataset(self.dataset_path, train_transform=self.transform)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        result = []
        dist_result = []
        labels_result = []

        print(f"Evaluation Start...")
        for data in test_dataloader:
            imgs1, imgs1_with_labels, imgs2, imgs2_with_labels, labels = data
            imgs1, imgs2, labels = imgs1.to(self.device), imgs2.to(self.device), labels.to(self.device)

            outputs1 = self.model(imgs1)
            outputs2 = self.model(imgs2)
            distances = F.pairwise_distance(outputs1, outputs2, keepdim=True)

            distances = distances.detach().cpu().numpy() * 1e2
            labels = np.expand_dims(labels.detach().cpu().numpy(), axis=-1)

            # 거리 정보, same/different 라벨 정보 따로 가짐
            dist_result = np.append(dist_result, distances)
            labels_result = np.append(labels_result, labels)

        # Result를 다 얻었으면, 그림을 그려보기
        print(max(dist_result), min(dist_result))
        print("make histogram...")
        bins = np.arange(np.min(dist_result) - 1., np.max(dist_result) + 1., 1)


        same_list = []
        different_list = []
        for idx in range(len(labels_result)):
            if labels_result[idx] == 0:
                same_list = np.append(same_list, dist_result[idx])
            else:
                different_list = np.append(different_list, dist_result[idx])

        same_hist, bins = np.histogram(same_list, bins)
        diff_hist, bins = np.histogram(different_list, bins)

        # Find best threshold
        max_distance = int(np.max(dist_result))
        max_threshold = 0
        max_F1_score = 0
        for temp_threshold in range(max_distance):
            TP, TN, FP, FN = 0, 0, 0, 0
            for dist in same_list:
                if dist <= temp_threshold:TP += 1
                else:FN += 1
            for dist in different_list:
                if dist >= temp_threshold:TN += 1
                else:FP += 1

            F1_score = (2 * TP) / (2 * TP + FP + FN)

            # print(f"f1 score: {F1_score} / temp_threshold: {temp_threshold}")
            if max_F1_score <= F1_score:
                max_F1_score = F1_score
                max_threshold = temp_threshold
        print(f"The best threshold: {max_threshold}")
        self.best_threshold = max_threshold

        if show_plot:
            print("Draw histogram...")
            plt.hist(same_list, bins, rwidth=0.5, color='green', alpha=0.5, label='Same')
            plt.hist(different_list, bins, rwidth=0.5, color='red', alpha=0.5, label='Different')
            plt.xlabel('Distance', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

        print("\n-------------------\n")




if __name__ == "__main__":
    pass










