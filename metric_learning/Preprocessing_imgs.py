'''
    이미지에 augmentation을 진행하여 따로 저장
    즉, origin image + augmentated image => 총 20개의 images
    이를 사용할 예정
    나중에 자동화를 진행하기 위해서 각 파일을 class화 시켜 메소드를 사용할 수 있도록 구현
    cv
'''

import os
import random

import torch
from torchvision import transforms
from PIL import Image


def preprocess_img(fixed_random_seed = True, # Random Seed을 0으로 계속 고정할건지 말건지를 선별
                   total_img_nums = 20,      # 인당 전처리할 이미지들 총 개수
                   ):

    if fixed_random_seed: torch.manual_seed(0)


    transform = transforms.Compose([
        # 이미지를 Rotation
        transforms.RandomRotation(15), # expand=True는 나중에
        # 밝기 변화
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        # 이미지를 (150, 150)로 맞추기 (원래는 (224, 224)가 기준)
        transforms.Resize((150,150)),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.1)
    ])


    # 임원진들 있는 폴더 경로 가져오기
    # imgs_dir='C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning'

    # 원본 이미지가 있는 경로
    origin_path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\test_images\\BARAM_people'
    # 전처리한 이미지를 저장할 경로
    preprocessed_path = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'

    # 원본 사람들 폴더 가져와서 리스트로 나열
    origin_imgs_dir = os.listdir(origin_path)
    # print(origin_imgs_dir)
    new_imgs_dir = os.listdir(preprocessed_path)
    # print(new_imgs_dir)

    # 이미 있었던 사람 얼굴 경로에 대해서는 제외하기
    temp = []
    for origin_img_dir in origin_imgs_dir:
        if origin_img_dir not in new_imgs_dir:
            temp.append(origin_img_dir)
    origin_imgs_dir = temp
    # print(f"입력할 사람 이름:{origin_imgs_dir}")

    # 이미지를 넣은 temp 공간 만들기
    img_dataset = []

    # 임원진들의 이미지들을 넣는 것을 iterate
    cnt=0

    for a_img_path in origin_imgs_dir:
        # 임원진들 내부의 이미지 경로 리스트로 반환
        a_class_imgs_paths = os.listdir(f"{origin_path}\\{a_img_path}")

        # 각 임원진들의 폴더의 이미지들에 접근
        origin_imgs = [Image.open(f"{origin_path}\\{a_img_path}\\{img_path}") for img_path in a_class_imgs_paths]

        for img_count in range(total_img_nums): # total_img_nums만큼 iterate
            random_sample_origin_img = random.choice(origin_imgs)
            preprocessed_img = transform(random_sample_origin_img)

            # 전처리된 이미지들을 담을 경로가 없다면
            if not os.path.exists(preprocessed_path):
                os.mkdir(preprocessed_path)

            # 각각의 전처리 이미지 label을 담을 경로가 없다면
            '''
                cnt만으로 진행해도 되는지에 대해서는 좀 더 알아가기
            '''
            if not os.path.exists(f"{preprocessed_path}\\{a_img_path}"):
                os.mkdir(f"{preprocessed_path}\\{a_img_path}")

            preprocessed_img.save(f'{preprocessed_path}\\{a_img_path}\\img_{img_count}.png', "png")

        cnt += 1




