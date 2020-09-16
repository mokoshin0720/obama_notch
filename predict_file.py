import os
from tensorflow import keras
from flask import Flask, request, redirect, url_for # フォームから送信した情報を扱う / ページの移動 / アドレス遷移
from flask import flash
from werkzeug.utils import secure_filename # ファイル名をチェックする
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
import sys, os
import numpy as np
import cv2
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

names = ["オバマ大統領","ノッチ"]
num_names = len(names)
img_size = 50

cascade_path = "face.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

UPRLOAD_FOLDER = "./upload" # 画像のアップロード先のディレクトリ
ALLOWED_EXTENSIONS = set(["png", "jpg", "gif"]) # アップロードされる拡張子の制限

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPRLOAD_FOLDER
app.secret_key = "super secret key"

def allowed_file(filename):
    # .があるかと拡張子が正しいのか確認　→　正しければ1, ダメなら0を返す
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def build_model():
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
    vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # VGG16の図の青色の部分は重みを固定（frozen）
    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    vgg_model.compile(loss='categorical_crossentropy',
            optimizer="sgd",
            metrics=['accuracy'])

    vgg_model.load_weights("./obama_last.h5")

    return vgg_model

@app.route("/", methods = ["GET", "POST"]) # ファイルを受け取る方法の指定
def upload_file():
    if request.method == "POST": # リクエストがPOSTかどうか

        if "file" not in request.files: # ファイルがなかった場合
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"] # データを取り出す

        if file.filename == "": # ファイル名がなかった場合
            flash("ファイルがありません")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # 危険な文字を削除（サニタイズ処理）
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename) # ファイルがあるパス
            file.save(filepath) # ファイルの保存

            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if img is None:
                return "顔を検出できません"
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = faceCascade.detectMultiScale(gray, 1.1, 3)

                if len(face) > 0:
            
                    for rect in face:

                        x = rect[0]
                        y = rect[1]
                        w = rect[2]
                        h = rect[3]

                        img = img[y:y+h, x:x+w]
                        filepath_face = os.path.join(app.config["UPLOAD_FOLDER"], "face_" + filename) # ファイルがあるパス
                        cv2.imwrite(filepath_face, img)

                else:
                    return "顔を検出できません"

            image = Image.open(filepath_face)
            image = image.convert('RGB')
            image = image.resize((img_size, img_size))
            data = np.asarray(image)
            
            X = []
            X.append(data)
            X = np.array(X)

            model = build_model()

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = str(result[predicted] * 100)

            return percentage + "%" + names[predicted] + "でしょう！"

    return '''

    <!doctype html>
    <html><head><title>ファイルをアップロードして判定</title>
    <meta charset="UTF-8">
    </head>
    <body>
    <h1>ファイルをアップロードして判定</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type = file name = file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''

from flask import send_from_directory

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8000")