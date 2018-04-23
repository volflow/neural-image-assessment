csv_path = '/home/ubuntu/data/ib-urls/' #'/Users/valentinwolf/data/ib-urls-test/'
CHUNK_SIZE = 1024
BATCH_SIZE = 64
MAX_WORKERS = 12
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
from multiprocessing.dummy import Pool

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
    downloads image from url; resizes image to target_size; returns image
    returns -1 if download was not successfull
    TODO: find pythonic way to write try: with:
    """
    try:
        with req.urlopen(url) as downloaded_img:
            try:
                with Image.open(downloaded_img) as img:
                    return resize(img,target_size)
            except OSError:
                print('\nfailed to download {}'.format(url))
                return -1

    except ValueError:
        print('\nunkown url type, failed open {}'.format(url))
        return -1
    except urllib.error.URLError:
        print('\nURLError, maybe server down, trying again in 10s: {}'.format(url))
        time.sleep(10)
        return download_img(url,target_size=target_size)
        

def inference_batchwise(model,batch):
    x = preprocess_input2(batch)
    scores = model.predict(x, batch_size=len(batch), verbose=0)
    return scores

def gen_batches(li, batch_size):
    """
    returns list of lists of size batch_size, with the elemnts of li
    if len(list) is not divisible by n the last list may be smaller than
    batch_size
    TODO: make this a generator (yield)
    """
    return [li[i:i+batch_size] for i in range(0, len(li), batch_size)]

def inference_from_urls(model,imgs,batch_size=32):
    target_size = (224, 224)
    with tf.device('/CPU:0'):
        total_imgs = len(imgs)
        score_list = []
        total_predicted = 0

        print('\rPredicting: {}/{} '.format(
                0,total_imgs), end='')

        for batch in gen_batches(imgs,batch_size=batch_size):
            x = np.zeros((len(batch), 224, 224, 3))

            download_start = time.time()
            max_workers = MAX_WORKERS
            with Pool(max_workers) as p:
                image_list = p.map(download_img,batch)

                failed_imgs = []
                # TODO: vectorize the loop
                for i,img in enumerate(image_list):
                    if img != -1:
                        x[i] = image_list[i]
                    else:
                        failed_imgs.append(i)

            download_time = (time.time() - download_start) / len(batch)
            inference_start = time.time()
            scores = inference_batchwise(model,x)

            # incicate failed predictions by score of [-1,0,0,...] => mean -1
            scores[failed_imgs] = 0
            scores[failed_imgs,0] = -1

            del x
            inference_time = (time.time() - inference_start) / len(batch)
            score_list.append(scores)
            total_predicted += len(batch)
            print('\rPredicting: {}/{} Download: {:.2f}s Inference {:.2f}s'.format(
                total_predicted,total_imgs,download_time, inference_time), end='')
    return np.vstack(score_list)

if __name__ == "__main__":
    # load network
    print('loading network')
    import evaluate
    model = evaluate.nasnet()
    csv_file_paths = sorted(glob.glob(csv_path+'/*.csv'))
    print(csv_file_paths)
    total_predicted = 0
    for csv_file_path in csv_file_paths:
        print("Reading new file: " + csv_file_path)

        csv_file = pd.read_csv(csv_file_path,sep=';',header=None,names=['id','url','score'])
        csv_file = csv_file.fillna(0.) # set nan values in score col to 0
        rows_in_file = csv_file.shape[0]
        # find first row with score 0
        index_list = csv_file.index[csv_file['score']==0.]
        if len(index_list) == 0:
            print("everything already predicted")
            total_predicted += rows_in_file
            continue

        chunk_size = CHUNK_SIZE
        offset = index_list[0]
        total_predicted += offset
        urls_list = csv_file['url'].tolist()[offset:]
        total_urls = len(urls_list)
        print("Found {} urls with score 0. Predicting ...".format(total_urls))
        for chunk in gen_batches(urls_list, chunk_size):
            scores = inference_from_urls(model,chunk,batch_size=BATCH_SIZE)
            mean_scores = mean_score(scores)
            csv_file.loc[offset:offset+len(chunk)-1,'score'] = mean_scores
            offset += len(chunk)
            total_predicted += len(chunk)
            print("\n{}/{} | Total: {} Saving to {}".format(offset,rows_in_file,total_predicted,csv_file_path))
            csv_file.to_csv(csv_file_path, sep=';', header=False)
