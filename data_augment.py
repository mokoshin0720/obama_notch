from PIL import Image,ImageEnhance
import glob
import numpy as np

names = ["obama", "notch"]
num_names = len(names)
img_size = 50
num_traindata = 120
num_testdata = 30

# 画像の読み込み

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, name in enumerate(names):
    # 各ディレクトリに含まれる画像を全てfilesに格納
    photos_dir = "./" + name + "_face"
    files = glob.glob(photos_dir + "/*.jpg")

    for i, file in enumerate(files):
        if i >= num_traindata + num_testdata: break
        img = Image.open(file)
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        data = np.asarray(img)

        if i < num_testdata: # テストデータは水増しせず、そのまま追加
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-30, 30, 3):
                # 5度刻みに回転をさせる
                img_r = img.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 左右を反転させる
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

                # コントラストを上げる
                img_precont = ImageEnhance.Contrast(img_r)
                img_cont = img_precont.enhance(2.0)
                data = np.asarray(img_cont)
                X_train.append(data)
                Y_train.append(index)

                # 明度を上げる
                img_prebri = ImageEnhance.Brightness(img_r)
                img_bri = img_prebri.enhance(2.0)
                X_train.append(data)
                Y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./obama_augmented_cont_bri.npy", xy)