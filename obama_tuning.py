import keras
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.applications import VGG16
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

names = ["obama", "notch"]
num_names = len(names)
img_size = 50

def main():
    X_train, X_test, y_train, y_test = np.load("./obama_augmented_contrast.npy", allow_pickle = True)
    
    # データの正規化
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255

    # 1-hotベクトル化（正解を1、間違いを0にする）
    y_train = np_utils.to_categorical(y_train, num_names)
    y_test = np_utils.to_categorical(y_test, num_names)

    model = model_train(X_train, y_train, X_test, y_test)
    model_evaluate(model, X_test, y_test)

def model_train(X_train, y_train, X_test, y_test):
    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_size, img_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # VGG16の図の緑色の部分（FC層）の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_names, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成（完成図が上の図）
    vgg_model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # VGG16の図の青色の部分は重みを固定（frozen）
    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    vgg_model.compile(loss='categorical_crossentropy',
            optimizer="sgd",
            metrics=['accuracy'])

    # vgg_model.summary()

    early_stopping =  EarlyStopping(monitor="val_accuracy", min_delta=0.00001, patience=5)

    history = vgg_model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[early_stopping], validation_data=(X_test, y_test))

    plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
    plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
    plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
    plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    #Final.pngという名前で、結果を保存
    plt.savefig('Final.png')
    plt.show()

    # モデルの保存
    vgg_model.save_weights("./obama_cnn.h5")

    return vgg_model

def model_evaluate(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print("loss:", scores[0])
    print("accuracy:", scores[1])

if __name__ == "__main__":
    main()