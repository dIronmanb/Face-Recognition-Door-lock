import cv2
import face_recognition
from time import time, sleep
import os
import copy

'''
name = 0    # label
cnt = 0     # 한 사람의 이미지에 시퀀스를 부여하기 위한 변수
maximum = 5 # 이미지 가져오는 개수
front_face_flag = False
capture = cv2.VideoCapture(0)
start_time = time()
'''
global_t = time()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return 0
    except OSError:
        print('Error: Creating directory. ' + directory)
        return 1


def time_interval(t=0.3):
    global global_t
    # print(time() - global_t)
    if time() - global_t > t:
        global_t = time()
        return 1
    else:
        return 0


def get_image_from_cv2(name=0, # label
                       cnt=0,  # 한 사람의 이미지에 시퀀스를 부여하기 위한 변수
                       maximum=5, # 한 사람 당 가져오는 이미지 개수
                       is_training_image=True,
                       ):
    front_face_flag = False  # 5초 대기하기 위한 변수
    start_time = time()

    if is_training_image:
        # 현재 경로에 이미지 라벨이 서로 겹치지 않도록 코드 짜기
        image_folder =  'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\test_images\\BARAM_people'
        file_list = os.listdir(image_folder)
        while str(name) in file_list:
            name += 1

        # 현존하는 디렉토리에 접근한다면 Error 발생하면서, 함수 종료
        error = createFolder(f"{image_folder}\\{str(name)}")
        if error:
            return
    else:
        pass

    # OpenCV 실행
    capture = cv2.VideoCapture(0)
    while cv2.waitKey(10) != ord('q'): #
        success, img = capture.read()
        # RGB로 변환
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # face_recognition.face_location으로 얼굴 crop
        faceLoc = face_recognition.face_locations(imgS)

        # 5초 잠시 대기: 카메라에 얼굴에 비치도록 다가가는 시간 필요
        if not front_face_flag:
            if time() - start_time >= 5.0:
                # flag 전환
                front_face_flag = True

        # 5초가 지난 이후에 코드
        if faceLoc and front_face_flag:
            # faceLoc[0]에 있는 x, y, h, w 정보 대입
            faceLoc = faceLoc[0]
            y1,x2,y2,x1 = faceLoc
            # origin img에서 x, y, h, w에 따라서 얼굴만 crop
            copied_img = copy.deepcopy(img)
            cropped_img = copy.deepcopy(img[faceLoc[0]:faceLoc[2], faceLoc[3]:faceLoc[1]])


            # 사각형 띄우기
            img = cv2.rectangle(img ,(x1,y1), (x2,y2), (0,0,255), 2)
            copied_img = cv2.rectangle(copied_img, (x1,y1), (x2,y2), (0,0,255), 2)

            # cropped image의 height 및 width가 150pixels보다 크면 저장
            if cropped_img.shape[0] > 150 and cropped_img.shape[1] > 150:
                cv2.putText(img, "Good!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), thickness=2)
                if is_training_image:
                    if time_interval(t=0.6):
                        cv2.imwrite(f'C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\test_images\\BARAM_people\\{str(name)}\\sample_{cnt}.jpg', cropped_img)
                        cnt += 1
                        if cnt >= maximum:
                            break
                else:
                    if cropped_img is not None:
                        return cropped_img, cv2.cvtColor(copied_img, cv2.COLOR_RGB2BGR)


            # 그게 아니면 Cropped image가 작다고 출력
            else:
                cv2.putText(img, "It's too small", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
        else:
            cv2.putText(img, "No Face!", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)

        # original 이미지 띄움
        '''
            나중에 내 얼굴이 잘 crop되고 있는지를 보여주기 위해
            OepnCV를 활용하여 정사각형 띄우기
        '''
        cv2.imshow("VideoFrame", img)


    capture.release()
    # OpenCV 종료
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    result = get_image_from_cv2(maximum=10, is_training_image=False)
    plt.imshow(result[1])
    plt.show()