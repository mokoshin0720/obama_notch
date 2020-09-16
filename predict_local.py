# ターミナルで動かすためのファイル
# predict_file.pyとほとんど同じ内容

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
import keras, sys, os
import numpy as np
import cv2
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

names = ["obama", "notch"]
num_names = len(names)
img_size = 50

cascade_path = "face.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

def build_model(): # モデルの構築 → obama_tuning.pyとほぼ同じ
    input_tensor = Input(shape=(img_size, img_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_names, activation='softmax'))

    vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    vgg_model.compile(loss='categorical_crossentropy',
            optimizer="sgd",
            metrics=['accuracy'])

    vgg_model.load_weights("./obama_cnn.h5")

    return vgg_model

def main():
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR) # 画像をimgに格納

    if img is None: # 画像がない場合
        print(sys.argv[1], "は顔を検出できませんでした。")
    else: # 顔を切り取る処理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.1, 3)

        if len(face) > 0:
            
            for rect in face:

                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

                img = img[y:y+h, x:x+w]
                cv2.imwrite("./" + sys.argv[2] + "_face.jpg", img)

        else: # 顔が読み取れない場合
            print(sys.argv[1],  "は顔を検出できませんでした。")

    # 切り取れた顔画像をnumpy配列に変換 → dataに格納
    img_test = Image.open("./" + sys.argv[2] + "_face.jpg")
    img_test = img_test.convert("RGB")
    img_test = img_test.resize((img_size, img_size))
    data = np.asarray(img_test) / 255
    
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()
    
    # 画像の予想
    result = model.predict([X])[0]
    print("result:", result)
    predicted = result.argmax()
    print(predicted)
    percentage = int(result[predicted] * 100)
    print(percentage + "%" + predicted[name] + "でしょう！")

if __name__ == "__main__":
    main()
