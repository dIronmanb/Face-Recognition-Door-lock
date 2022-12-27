'''DataLoader 부분'''
import operator

import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from itertools import combinations, product, groupby
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
torch.manual_seed(0)
expansion = 1

class CustomDataset(Dataset):
    def __init__(self, img_dir, train_transform=None, target_transform=None):
        '''
        :param img_dir:
            임원진들의 이름이 적혀진 폴더가 각각 존재
            임원진 폴더 내에는 20개의 이미지가 존재
            (5개는 미리 찍고, 나머지 15개는 따로 생성해낸다.)
        :param transform:
            image transformation
        :param target_transform:
            labels target_transformation
            다만, labels는 same pair label, different pair label이므로
            굳이 transformation을 사용할 필요 X
        '''

        dataset = ImageFolder(img_dir, transform=transforms.Resize((150,150)))

        self.same_pair = []
        self.differ_pair = []
        self.total_pair = None
        self.train_transform = train_transform
        self.target_transform = target_transform
        img_dataset = []
        imgs_dir = os.listdir(img_dir) # 각 임원진들의 이름만 담긴다.


        # 각 임원진들의 폴더 접근 (It works)
        '''
            한 임원진 당 가진 이미지 개수: 20
            현재 임원진 클래스 개수: 7
        '''
        # for a_img_path in imgs_dir:
        #     # 임원진들 내부의 이미지 경로 리스트로 반환
        #     a_class_imgs_paths = os.listdir(f"{img_dir}\\{a_img_path}")
        #     # 각 임원진들의 폴더의 이미지들에 접근
        #     a_class_imgs = [Image.open(f"{img_dir}\\{a_img_path}\\{img_path}").resize((150,150)) for img_path in a_class_imgs_paths]
        #     img_dataset.append(a_class_imgs)
        #     # img_dataset[0], img_dataet[1], ... , img_dataset[n]

        dataset = groupby(dataset, key=lambda x: x[1])

        for key, grouped_data in dataset:
            img_dataset.append(list(grouped_data))
        # img_dataset[num_class][num_imgs][img or label]


        # intra-class에 대하여 nC2 사용 (It works)
        '''
            20 C 2 = 20 * 19 / 2 = 190개의 조합
            7 * 190 = 1330
        '''
        for one_class_data in img_dataset:
            two_same_image_bundle = list(combinations(one_class_data, 2)) # 한 클래스에 대한 이미지들 접근
            number_of_one_class_same_pair = len(two_same_image_bundle)
            for same_images in two_same_image_bundle:
                # sample을 이렇게 하나씩 얻어가기
                # iteration = 클래스 개수 * 이미지 개수
                self.same_pair.append((*same_images, 0))

        # inter-class에 대하여 각각을 사용
        # 클래스의 개수만큼을 iterate
        for index in range(len(img_dataset)):
            # 자기 자신을 제외한 다른 클래스 불러오기
            other_classes = [img_dataset[i] for i in range(len(img_dataset)) if i != index]

            single_different_pair_temp = []

            # 다른 클래스들에 대해 임원진 클래스 하나씩 불러오기
            for another_class in other_classes:
                # 20개 * 20개 -> 400개의 prodcut
                products = list(product(img_dataset[index], another_class))

                for i in products:
                    single_different_pair_temp.append((*i, 1))

            # 400개 * 6 classes = 2400개의 different pair
            # print(len(single_different_pair_temp))
            # print(single_different_pair_temp[0][0].shape)
            # print(single_different_pair_temp[0][1].shape)
            # print(single_different_pair_temp[0][2])


            # 단일 different pair에 대해 랜덤 초이스
            # nC2 = X * 클래스 개수
            # X = int(nC2 / 클래스 개수)

            self.differ_pair = self.differ_pair + random.sample(single_different_pair_temp, number_of_one_class_same_pair*expansion)

            '''다만, 같은 조합이 들어온다면???'''
            # 24 x (9-1) = 192
            # A,B 내의 이미지 20 x 20 = 400에서의 24개만 random choice
            # 나머지 클래스 9 - 1 = 8(명)

        # 이렇게 해서 (anchor, paired_img, label) 모두 끝났다.
        self.total_pair = self.same_pair + self.differ_pair
        random.shuffle(self.total_pair)

    def __len__(self):
        return len(self.total_pair)

    def __getitem__(self, idx):

        img1_with_label, img2_with_label, label = self.total_pair[idx]
        img1, img1_label = img1_with_label
        img2, img2_label = img2_with_label

        if self.train_transform:
            img1 = self.train_transform(img1)
            img2 = self.train_transform(img2)
        if self.target_transform:
            pass

        return img1, img1_label, img2, img2_label, label


# Quick Test
# Dataset 불러와서 Same인지 Different인지를 알아맞춘다.
# 문제 없음
if __name__ == "__main__":
    path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    train_dataset = CustomDataset(path)
    print(len(train_dataset))

    for i in range(len(train_dataset)):
        if i >= 2:
            break
        img1, img1_label, img2, img2_label, label = train_dataset[i]
        img1 = np.array(img1)
        img2 = np.array(img2)
        label = "Same" if label==0 else "Different"
        print(img1_label, img2_label, label)

        two_imgs = np.concatenate([img1, img2], axis=1)
        plt.imshow(two_imgs)
        plt.title(label)
        plt.show()
