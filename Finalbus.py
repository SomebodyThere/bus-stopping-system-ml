import cv2
import torch
import time
import datetime
import numpy as np
from PIL import ImageFont, ImageDraw, Image

#한글 출력
COLOR = (0, 0, 0)  # 흰색
FONT_SIZE = 30

def put_text(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('./NanumSquare_acEB.ttf', font_size) #경로 맞춰서 수정
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)


#사람 카운팅 및 문닫힘 시간 계산
start_time = None
closing_time = 30 #기본으로 설정된 문 닫히는 시간
p_count = 0


#yolo모델
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', #인식이 오래 걸리면 yolov5s로 바꿔도 됩니다! 바꾸면 인식이 살짝 안될 수 있음!
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')  # 예측 모델
yolo_model.classes = [0]  # 예측 클래스 (사람)

#웹캠 초기화 및 촬영데이터저장
cap = cv2.VideoCapture(0)
ret, frame = cap.read()



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #화면 좌우반전
    if frame is None:
        break

    results = yolo_model(frame)
    results_refine = results.pandas().xyxy[0].values
    nms_human = len(results_refine)
    rows, cols = frame.shape[:2]
    
    # 한글 텍스트를 이미지에 그리기
    frame = put_text(frame, '명이 탑승 중 입니다', (50, 50), FONT_SIZE, COLOR)
    frame = put_text(frame, '초 뒤에 문이 닫힙니다', (70, 100), FONT_SIZE, COLOR)
    

    #사람 수 카운팅에 대해 시간 계산
    if nms_human >= 0:
        p_count = nms_human
        if start_time is None :
            start_time = time.time()
        
        #문닫힘타이머는 최대 60초, 한명이 탑승할 때마다 기존 시간에 15초씩 추가되게 작성 
        elapsed_time =  int(closing_time + (p_count*15)  - (time.time() - start_time))
        if elapsed_time >= 60:
            elapsed_time = 60
            
        if elapsed_time <= 0:
            elapsed_time = 0 #남은시간이 0초가 되면 문닫힘 문구 출력
            frame = put_text(frame, '문이 닫힙니다 안전에 유의하세요', (200, 200), FONT_SIZE, COLOR)
        
        #인식한 사람 박스처리
        for bbox in results_refine:
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))

            frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)
            
    cv2.putText(frame, str(p_count) , (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, str(elapsed_time),(20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    
    # 프로그램 시작 시간 (문 열림 시간) 출력
    start_struct_time = time.localtime(start_time)
    display_time = time.strftime('%H:%M:%S',time.localtime(start_time))
    cv2.putText(frame, "open: " + display_time, (cols-300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
        
    cv2.imshow("bus", frame)
    
    if cv2.waitKey(1) == ord("r"): #타이머리셋 
        start_time = time.time()
        p_count = 0
    
    
    if cv2.waitKey(1) == ord("q"): #창닫기
        break



cap.release()
cv2.destroyAllWindows()    