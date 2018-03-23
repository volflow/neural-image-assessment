from flask import Flask
from utils.downloader import download_img,download_thumbnail
import sys
from evaluate import evaluate_mobilenet
import os

app = Flask(__name__)
fn = "temp/flask_test.jpg"

@app.route('/url=<path:url>')
def index(url):
    download_img(url,fn)
    download_thumbnail(url,fn+"thumbnail.jpg")
    pred = evaluate_mobilenet([fn,fn+"thumbnail.jpg"])
    return str(pred)
