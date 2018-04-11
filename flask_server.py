from flask import Flask, render_template, request, abort
from utils.downloader import download_img,download_thumbnail
import sys
from evaluate import evaluate, mobilenet, nasnet
import os
import json

app = Flask(__name__)
fn = "temp/flask_test.jpg"
model = mobilenet()
model._make_predict_function()
#model2 = nasnet()
#model2._make_predict_function()
test = "teeest"

@app.route('/',methods=['GET'])
def index():
    test = "teeest"
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    _url = request.form['imgUrl']
    print(_url)

    try:
        download_img(_url,fn)
        #download_thumbnail(url,fn+"thumbnail.jpg")
        _fn,_mean,_std = evaluate(model,[fn])[0]
        print(_fn,_mean,_std)
    except ValueError as err:
            print(err)
            abort(400)
            return json.dumps({'error':str(err)})

    return json.dumps({'filename':_fn,'mean': _mean, 'std':_std})#render_template('index.html', foo=False,test="123")



@app.route('/url=<path:url>')
def result(url):
    download_img(url,fn)
    download_thumbnail(url,fn+"thumbnail.jpg")

    pred = evaluate(model,[fn,fn+"thumbnail.jpg"])
    #pred2 = evaluate(model2,[fn,fn+"thumbnail.jpg"])
    return 'Mobilenet: ' + str(pred) #+ '| Nasnet: ' + str(pred2)

if __name__ == '__main__':
    app.run(debug=True)
