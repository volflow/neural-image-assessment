from flask import Flask
import utils
import sys
sys.path.append("./neural-image-assessment")
import eval_mobilenet
import os

app = Flask(__name__)
fn = "flask_test.jpg"

@app.route('/url=<path:url>')
def index(url):
    utils.download_img(url,fn)
    utils.download_thumbnail(url,fn+"thumbnail.jpg")
    #print(os.listdir())
    pred1 = eval_mobilenet.evaluate([fn,fn+"thumbnail.jpg"])
    return str(pred1)
