# TODO: Flow 설명
#   ---등록----
#   1. face_detect_only_opencv.py 및 카메라로 이미지 수집
#   2. 폴더로 저장된 이미지를 Preprocessing_imgs.py로 수집한 이미지를 증강함
#   3.1 전체 데이터셋에 대해서 resnext50을 main.py에서 학습 진행 (Metric_Learning_DataLoader.py도 같이 확인하기)
#        이때 epoch은 10 전후
#   3.2. preprocessed 이미지를 extract_vector.py를 사용하여 feature vector 저장
#   ---검증----
#   4.1 get_image.py 및 카메라로 이미지 수집 (이때의 이미지 경로는 Verfication)에 저장
#   4.2 또는, get_image.py로 이미지 받아오고, resnext50에 바로 넘겨서 feature vector 뽑음
#   5. extract_vector.py를 통해 이전에 저장한 feature vector를 가져와서 비교
#   6. 일정 threshold보다 작으면 accepted, 크면 denied
from metric_learning.face_detect_only_opencv import get_image_from_cv2
from metric_learning.Preprocessing_imgs import preprocess_img
from get_model import resnext50
from main import ModelManager
from extract_vector import FeatureVectorManager

import torch
import os
from os.path import exists
from datetime import datetime

import matplotlib.pyplot as plt

import serial
import time
import cv2

def send_signal(signal, port):

    py_serial = serial.Serial(
        # Window
        port=port,
        # 보드 레이트 (통신 속도)
        baudrate=9600,
    )

    if signal:
        # denied
        commend = "X"
    else:
        # accepted
        commend = "O"

    py_serial.write(commend.encode())
    time.sleep(0.1)
    if py_serial.readable():
        # 들어온 값이 있으면 값을 한 줄 읽음 (BYTE 단위로 받은 상태)
        # BYTE 단위로 받은 response 모습 : b'\xec\x97\x86\xec\x9d\x8c\r\n'
        response = py_serial.readline()
        # 디코딩 후, 출력 (가장 끝의 \n을 없애주기위해 슬라이싱 사용)
        print(response[:len(response) - 1].decode())

    print("Sending message Successfully.")

def pipeline(get_image=True, preprocessing_img=True, verification=False):
    # 사람 얼굴 이미지를 가져와서 저장
    if get_image:
        get_image_from_cv2(name=0, maximum=10)

    # 사람 얼굴 이미지를 전처리
    if preprocessing_img:
        preprocess_img(fixed_random_seed=True, total_img_nums=24)

    # 모델 load, training, and save
    # SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss\\resnet50_2022-10-13_23-1-12_50.pt'
    SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    # TODO: 최근에 학습한 모델 load
    LOAD_PATH = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    LOAD_PATH = f"{LOAD_PATH}\\{os.listdir(LOAD_PATH)[-1]}"
    DATASET_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    training_data_path = "C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs"
    model = resnext50()

    # model_manager
    model_manager = ModelManager(save_path=SAVE_PATH,
                                 load_path=LOAD_PATH,
                                 model=model,
                                 is_weight_loaded=True,
                                 epochs=5,
                                 num_of_class=len(os.listdir(training_data_path)))


    # Get Dataset
    model_manager.get_dataset(DATASET_PATH, batch_size=32, num_workers=4)
    # Train model
    model_manager.train()
    # Evaluate model
    model_manager.evaluate()
    best_threshold = model_manager.best_threshold
    # Save model and plot result
    model_manager.save_model()


    # 학습된 backbone을 통해 feature vector 추출하기
    people_img_dir =  'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    x = datetime.now()

    hour, minute, second = x.hour, x.minute, x.second
    if hour < 10:
        hour = '0' + str(x.hour)
    if minute < 10:
        minute = '0' + str(x.minute)
    if second < 10:
        second = '0' + str(x.second)

    new_directory = f"\\{x.year}_{x.month}_{x.day}"
    now_time = f"{hour}-{minute}-{second}"

    original_saved_path = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\extracted_vectors'

    if not os.path.isdir(original_saved_path+new_directory):
        os.mkdir(original_saved_path+new_directory)
    database_name = f'\\{now_time}.json'

    model_manager.model.eval()
    # model_manager.model ..???

    # feature vector manager로 fv 관리하기
    fv_manager = FeatureVectorManager(people_img_dir, model_manager.model, best_threshold)
    fv_manager.extract_vectors()
    fv_manager.save_vectors(saved_path=original_saved_path + new_directory + database_name)

    # 검증하기
    if verification:
        # 1. fv_manager에서 database를 load하기
        fv_manager.load_csv_file(load_path=original_saved_path + new_directory + database_name)

        # 2. 이미지 가져오기
        cropped_img, origin_img = get_image_from_cv2(is_training_image=False)
        plt.imshow(origin_img)
        plt.show()

        # 3. database를 통해 accepted/denied 판단하기
        state = fv_manager.permit(cropped_img)

        if state:
            cv2.putText(origin_img, "Accepted!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
        else:
            cv2.putText(origin_img, "Denied!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)
        plt.imshow(origin_img)
        plt.show()


def verificate(show_plot=True):
    # SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss\\resnet50_2022-10-13_23-1-12_50.pt'
    SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    # TODO: 최근에 학습한 모델 load
    LOAD_PATH = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    LOAD_PATH = f"{LOAD_PATH}\\{os.listdir(LOAD_PATH)[-1]}"

    DATASET_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    training_data_path = "C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs"
    model = resnext50()

    # model_manager
    model_manager = ModelManager(save_path=SAVE_PATH,
                                 load_path=LOAD_PATH,
                                 model=model,
                                 is_weight_loaded=True,
                                 epochs=3,
                                 num_of_class=len(os.listdir(training_data_path)))

    # Get Dataset
    model_manager.get_dataset(DATASET_PATH, batch_size=32, num_workers=4)
    # Evaluate model
    model_manager.evaluate(show_plot=show_plot)
    best_threshold = model_manager.best_threshold


    # 학습된 backbone을 통해 feature vector 추출하기
    people_img_dir = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    x = datetime.now()
    original_saved_path = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\extracted_vectors'
    original_saved_path = f"{original_saved_path}\\{sorted(os.listdir(original_saved_path))[-1]}"
    original_saved_path = f"{original_saved_path}\\{sorted(os.listdir(original_saved_path))[-1]}"
    print(original_saved_path)

    # feature vector manager로 fv 관리하기
    fv_manager = FeatureVectorManager(people_img_dir, model_manager.model.eval(), best_threshold)

    # 1. fv_manager에서 database를 load하기
    fv_manager.load_csv_file(load_path=original_saved_path)

    # 2. 이미지 가져오기
    cropped_img, origin_img = get_image_from_cv2(is_training_image=False)
    plt.imshow(origin_img)
    plt.show()

    # 3. database를 통해 accepted/denied 판단하기
    state = fv_manager.permit(cropped_img)

    if state:
        cv2.putText(origin_img, "Accepted!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
    else:
        cv2.putText(origin_img, "Denied!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)
    plt.imshow(origin_img)
    plt.show()

def extract_featue_vector(show_plot=True):
    # 얼굴 이미지 등록
    get_image_from_cv2(name=0, maximum=10)

    preprocess_img(fixed_random_seed=True, total_img_nums=24)

    # SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss\\resnet50_2022-10-13_23-1-12_50.pt'
    SAVE_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    # TODO: 최근에 학습한 모델 load
    LOAD_PATH = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\saved_model_CE_and_Contrastive_loss'
    LOAD_PATH = f"{LOAD_PATH}\\{os.listdir(LOAD_PATH)[-1]}"
    DATASET_PATH = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    training_data_path = "C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs"
    model = resnext50()

    # model_manager
    model_manager = ModelManager(save_path=SAVE_PATH,
                                 load_path=LOAD_PATH,
                                 model=model,
                                 is_weight_loaded=True,
                                 epochs=3,
                                 num_of_class=len(os.listdir(training_data_path)))
    # Get Dataset
    model_manager.get_dataset(DATASET_PATH, batch_size=32, num_workers=4)
    # Evaluate model
    model_manager.evaluate(show_plot=show_plot)
    best_threshold = model_manager.best_threshold
    model_manager.model.eval()

    # 학습된 backbone을 통해 feature vector 추출하기
    people_img_dir = 'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\proprecessed_images\\preprocessed_BARAM_imgs'
    x = datetime.now()

    if x.hour < 10:
        hour = '0' + str(x.hour)
    if x.minute < 10:
        minute = '0' + str(x.minute)
    if x.second < 10:
        second = '0' + str(x.second)

    new_directory = f"\\{x.year}_{x.month}_{x.day}"
    now_time = f"{hour}-{minute}-{second}"

    original_saved_path = f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\metric_learning\\extracted_vectors'

    if not os.path.isdir(original_saved_path + new_directory):
        os.mkdir(original_saved_path + new_directory)
    database_name = f'\\{now_time}.json'

    fv_manager = FeatureVectorManager(people_img_dir, model_manager.model, best_threshold)
    fv_manager.extract_vectors()
    fv_manager.save_vectors(saved_path=original_saved_path + new_directory + database_name)



if __name__ == "__main__":
    # 학습 코드
    # pipeline(get_image=True, preprocessing_img=True, verification=False)
    # verification 코드
    verificate(show_plot=True)
    # feature vector 등록 코드
    # extract_featue_vector(show_plot=False)







