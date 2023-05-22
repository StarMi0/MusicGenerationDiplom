import time

from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, reqparse, api
from func import gan_melody
from func.generator import load_trained_model, gan_melody, save_pred



# создаем приложение
app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/')
def musical():
    return "Music Generation"


if __name__ == '__main__':
    app.run(debug=True)
