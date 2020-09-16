import cv2
import numpy as np
import os

cascade_path = "face.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

names = ["obama", "notch"]

for name in names:
    # 画像データのあるディレクトリ
    input_data_path = "./" + str(name)
    # 切り抜いた画像を保存するディレクトリ
    save_path = "./" + str(name) + "_face/"
    # 収集した画像の枚数
    img_num = 300
    # 顔検知に成功した数(デフォルト0)
    face_detect_num = 0

    for i in range(img_num):
        # 1枚ずつimgに画像を格納
        img = cv2.imread(input_data_path + '/0' + str(i) + ".jpg", cv2.IMREAD_COLOR)
        if img is None: # 画像がない場合
            print("img" + str(i) + "は顔を検出できませんでした。")
        else: # 顔の切り抜き処理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = faceCascade.detectMultiScale(gray, 1.1, 3)
            if len(face) > 0:
                for rect in face:

                    x = rect[0]
                    y = rect[1]
                    w = rect[2]
                    h = rect[3]
                    cv2.imwrite(save_path + "cutted_" + str(face_detect_num) + ".jpg", img[y:y+h, x:x+w])
                    face_detect_num += 1
            else: # 顔が読み取れない場合
                print("img" + str(i) + "は顔を検出できませんでした。")
