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
        img = cv2.imread(input_data_path + '/0' + str(i) + ".jpg", cv2.IMREAD_COLOR)
        if img is None:
            print("img" + str(i) + "は顔を検出できませんでした。")
        else:
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
            else:
                print("img" + str(i) + "は顔を検出できませんでした。")

print("顔画像の切り取り作業、正常に動作しました。")
print("検知した顔の数は" + str(face_detect_num) + "です")