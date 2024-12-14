# app.py
import os
import cv2
from flask import Flask, Response, render_template, request
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Загрузите модель YOLO
model = YOLO('C:\\Users\\lyami\\PycharmProjects\\FlaskProject1\\custom_yolov112\\weights\\best.pt')  # Укажите путь к вашим весам модели


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получаем файл из формы
        file = request.files['image']
        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
            # Сохраняем загруженное изображение
            img_path = os.path.join('static/uploads', file.filename)
            file.save(img_path)

            # Обнаружение объектов
            results = model(img_path)

            # Получите первый результат из списка
            detected_image = results[0]

            # Сохранение результатов
            detected_img_path = f'static/uploads/detected_{file.filename}'
            detected_image.save(detected_img_path)  # Предполагается, что метод save() доступен

            return render_template('result.html', detected_image=detected_img_path)

    return render_template('index.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    cap = cv2.VideoCapture(0)  # Используйте 0 для первой камеры
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Выполните детектирование
        results = model(frame)
        annotated_frame = results[0].plot()  # Предполагая, что метод plot() возвращает изображение с аннотациями

        # Кодирование изображения в JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/live')
def live():
    return render_template('live.html')
