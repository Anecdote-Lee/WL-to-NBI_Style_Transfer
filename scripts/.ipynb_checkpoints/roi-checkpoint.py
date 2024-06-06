import cv2
import numpy as np
import os



data_route = "/home/jslee/uvcgan2/data/new_crop_colonoscopic"
save_route = "/home/jslee/uvcgan2/data/roi_colonoscopic"
for mode1 in {"test", "train"}:
    for mode2 in {"nbi", "wl"}:
        real_route = data_route+"/"+mode1+"/" + mode2
        for image_ad in os.listdir(real_route):
            img = cv2.imread(real_route + "/" + image_ad)
            x,y,w,h = cv2.selectROI('img', img, False)
            if w and h:
                roi = img[y:y+h, x:x+w]
                cv2.imshow('cropped', roi)                   # ROI 지정 영역을 새창으로 표시
                cv2.moveWindow('cropped', 0, 0)              # 새창을 화면 측 상단으로 이동
                cv2.imwrite(f'{save_route}/{mode1}/{mode2}/roi_{img}', roi)  # ROI 영역만 파일로 저장
                
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()