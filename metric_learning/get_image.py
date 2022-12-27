'''OpenCV를 사용하여 임원진 얼굴을 받아오는 곳 Using Object Detection'''
from Object_Detection import get_bounding_box
import cv2
import os

webcam = cv2.VideoCapture(0)

get_image_cnt = 0
result_imgs = []
origin_images_path = "C:\\Users\\petnow\\PycharmProjects\\BARAM_03_02\\origin_images"
# origin_image에서 문자열로 폴더를 넣기 보다 숫자 하나씩 증가하는 방향으로 넣기
# (ex) person_00, person_01, person_02, ...
person_classes = os.listdir(origin_images_path)
if person_classes:
    name = "person_0"
else:
    character_num = person_classes[-1][-2:]
    name = f"person_{character_num+1}"

if not webcam.isOpened():
    print("Could not open webcam")
    exit()


while webcam.isOpened():
    status, frame = webcam.read()

    if status:
        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # elif 이미지를 다섯 개 얻어오면
    elif get_image_cnt >= 5:
        break

    # 이미지를 잘 얻어오면 get_image_cnt를 1 카운트
    state = get_bounding_box(frame)
    if state:
        result_imgs.append(state)
        get_image_cnt += 1

webcam.release()
cv2.destroyAllWindows()

img_cnt = 0
# 여기서는 얻은 이미지들에 대해 origin_images에 저장하기
for img in result_imgs:
    img.save(f"{origin_images_path}\\{name}_{img_cnt}.png", "png")
    img_cnt += 1


print("Getting images is finished!")
print(f"The number of images gotten by openCV is {img_cnt}")