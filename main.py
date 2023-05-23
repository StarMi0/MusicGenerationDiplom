import os

from flask import Flask, render_template, request, send_file
from func.generator import load_trained_model, gan_melody, save_pred
from func.functions import output_path

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение данных из формы
        length = float(request.form['length'])
        low_freq = int(request.form['low_freq'])
        high_freq = int(request.form['high_freq'])
        speed = float(request.form['speed'])
        print(length, low_freq, high_freq, speed)
        # Генерация MID файла
        model = load_trained_model("data/checkpoint_model_15.hdf5")
        save_pred(gan_melody(model))
        

        # Отправка файла для проигрывания
        return render_template('play.html', filename=os.path.join(output_path, "LSTM_music.mid"))

    # Отображение формы
    return render_template('index.html')


@app.route('/download')
def download():
    # Скачивание сгенерированного MID файла
    return send_file(os.path.join(output_path, "LSTM_music.mid"), as_attachment=True)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=8000)
