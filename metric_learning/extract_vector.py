# TODO: 모든 이미지들을 가져와서 featutre vector 추출하기
#   feature vector를 .csv 파일로 저장 (이때 한 행에 대한 정보는 (img_label, feature_vector)
#   근데 굳이 img_label이 필요할까? feature vector들 사이에서만 비교하면 되므로
#   일단은 img_label도 추가하자.
#   *주의* 새로 들어갈 데이터와 이미 저장된 데이터 사이에서 충돌이 없도록 하기
#   json file을 쓸건지, pandas.Dataframe을 쓸건지는 내가 정하기
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from get_model import resnext50


# import pandas as pd
import json
import numpy as np
import os


class FeatureVectorManager():

    def __init__(self, img_dir, model, best_threshold):
        self.img_dir = img_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.best_threshold = best_threshold
        self.dataset = ImageFolder(img_dir, transform=transforms.Compose([
           transforms.Resize((150, 150)),
           transforms.ToTensor()]))
        self.img_list = list()
        self.saved_path = None
        self.load_path = None

    def extract_vectors(self):
        print("Start extracting feature vectors...")
        for img, label in self.dataset:
            img = img.unsqueeze(0)
            img = img.to(self.device)
            feature_vector = self.model(img)
            self.img_list.append((label, feature_vector.tolist()))
        print(f"Extracting feature vectors finished: len(img_list): {len(self.img_list)}")

    def permit(self, image):
        if isinstance(image, np.ndarray):
            print(f"불러온 이미지의 dtype:{image.dtype}")
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                        transforms.Resize((150, 150)),
                        transforms.ToTensor(),
            ])
        image = transform(image)
        image = image.unsqueeze(dim=0)
        feature_vector = self.model(image.to(self.device)).detach().cpu().numpy()

        if self.load_path is not None:
            self.load_csv_file(self.load_path)
        else:
            print(f"Once implement '.load_csv_file()' method")
            return

        ########################################################################
        margin = -0.0
        for i in self.img_list:
            name, fv = i
            distance = np.linalg.norm(fv - feature_vector, ord=2, axis=None, keepdims=False)*1e2
            print(f"Distance between two feature vectors: {distance}")

            if distance < self.best_threshold + margin:
                print(f"Permission accepted (adjusted thershold: {self.best_threshold + margin} / distance: {distance})")
                return 1
        else:
            print(f"Permssion denied (adjusted thershold: {self.best_threshold + margin} / distance: {distance})")
            return 0
        ########################################################################


    def save_vectors(self, saved_path:str):
    # json file로 저장하기
        print("Strat saving feature vectors in json files...")
        self.saved_path = saved_path
        if not self.img_list:
            print("Database is empty!")
            return
        else:
            with open(self.saved_path, 'w', encoding='utf-8') as make_file:
                json.dump(self.img_list, make_file, indent=4)
        print("Saving feature vectors finished!")
        print("\n-------------------\n")

    def load_csv_file(self, load_path:str):
        self.load_path = load_path
        print("Strat loading feature vectors in json files...")
        with open(self.load_path, 'r') as f:
            self.img_list = json.load(f)
        print("Finished!")
        print("\n-------------------\n")
        # print(f"One example: self.img_list[0]'s label:{self.img_list[0][0]}\n"
        #       f"             self.img_list[0]'s image.shape:{np.array(self.img_list[0][1]).shape}")

if __name__ == "__main__":
    img_dir = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    original_saved_path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\extracted_vectors\\2022_11_03\\'
    database_name = 'extracted_feature_vectors_0.json'
    trained_model_path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss\\resnet50_2022-10-13_23-1-12_50.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnext50()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    data_saver = FeatureVectorManager(img_dir, model)
    # data_saver.extract_vectors()
    # data_saver.save_vectors(saved_path=original_saved_path+database_name)
    data_saver.load_csv_file(load_path=original_saved_path+database_name)

