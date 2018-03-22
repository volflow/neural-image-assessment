import urllib.request as req
from PIL import Image
import sys
sys.path.append("./neural-image-assessment")
import resize


def download_img(url, fn):
    """
    url: url of image
    fn: image will be saved as filename
    """
    def hook(chunk_number,max_size,total_size):
        if chunk_number == 0:
            print("Beginning download...")
    req.urlretrieve(url,fn,hook)
    print("Downloaded {} as {}".format(url,fn))
    return

def download_thumbnail(url,fn,target_size=(224,224)):
    img = Image.open(req.urlopen(url))
    resize.resize_from_obj(img,fn,target_size=target_size, keep_aspect_ratio=False, resample=Image.LANCZOS)
    return

