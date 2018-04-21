csv_path = '/home/ubuntu/data/ib-urls/' #'/Users/valentinwolf/data/ib-urls-test/'

import numpy as np
import pandas as pd
import glob

from utils.score_utils import mean_score

import urllib.request as req
from PIL import Image
from keras.preprocessing.image import img_to_array
from utils.nasnet import preprocess_input as preprocess_input2

import tensorflow as tf
import time
from multiprocessing import Pool

def resize(img,target_size):
    """
    resize images to target_size; keeps aspect_ratio; adds black padding at the bottom/right if necessary
    """
    resample_method = Image.LANCZOS

    img.thumbnail(target_size,resample_method)
    padding = Image.new('RGB',
                 target_size)
    padding.paste(img)
    return padding

def download_img(url,target_size=(224,224)):
    """
    downloads image from url; resizes image to target_size; retruns image
    """
    try:
        img = Image.open(req.urlopen(url))
    except OSError as inst:
        print('failed to download {}'.format(url))
        return -1
    img = resize(img,target_size)
    return img

def inference_batchwise(model,batch):
    x = preprocess_input2(batch)
    scores = model.predict(x, batch_size=len(batch), verbose=0)
    return scores


def inference_from_urls(model,imgs,batch_size=32):
    target_size = (224, 224)
    with tf.device('/CPU:0'):
        total_imgs = len(imgs)
        score_list = []
        batches = int(np.ceil(total_imgs / batch_size))

        print('\rEvaluating: {}/{} '.format(
                0,total_imgs), end='')

        for b in range(batches):
            batch = imgs[b*batch_size:(b+1)*batch_size]
            x = np.zeros((len(batch), 224, 224, 3))

            download_start = time.time()
            MAX_WORKERS = 14
            with Pool(MAX_WORKERS) as p:
                image_list = p.map(download_img,batch)
                
                failed_imgs = []
                # TODO: vectorize the loop
                for i,img in enumerate(image_list):
                    if img != -1:
                        x[i] = image_list[i]
                    else: 
                        failed_imgs.append(i)

            download_time = (time.time() - download_start) / batch_size
            inference_start = time.time()
            scores = inference_batchwise(model,x)
            scores[failed_imgs].fill(-1) 
            del x
            inference_time = (time.time() - inference_start) / batch_size
            score_list.append(scores)
            print('\rEvaluating: {}/{} Download: {:.2f}s Inference {:.2f}s'.format(
                b*batch_size+len(batch),total_imgs,download_time, inference_time), end='')

    return np.vstack(score_list)

if __name__ == "__main__":
    # load network
    print('loading network')
    import evaluate
    model = evaluate.nasnet()
    csv_file_paths = sorted(glob.glob(csv_path+'/*.csv'))
    print(csv_file_paths)
    for csv_file_path in csv_file_paths:
        print("Reading new file: " + csv_file_path)

        csv_file = pd.read_csv(csv_file_path,sep=';',header=None,names=['id','url','score'])
        csv_file = csv_file.fillna(0.) # set nan values in score col to 0

        # find first row with score 0
        index_list = csv_file.index[csv_file['score']==0.]
        if len(index_list) == 0: 
            print("everything already predicted")
            continue
        offset = index_list[0]

        urls_list = csv_file['url'].tolist()[offset:]
        total_urls = len(urls_list)
        print("Found {} urls with score 0. Predicting ...".format(total_urls))

        chunk_size = 64
        chunks = int(np.ceil(total_urls / chunk_size))
        for i in range(chunks):
            start_i = i*chunk_size
            end_i = (i+1)*chunk_size
            chunk = urls_list[start_i:end_i]
            scores = inference_from_urls(model,chunk,batch_size=64)
            mean_scores = mean_score(scores)
            csv_file.loc[start_i+offset:end_i+offset-1,'score'] = mean_scores
            print("\nSaving progress to {}".format(csv_file_path))
            csv_file.to_csv(csv_file_path, sep=';', header=False)
