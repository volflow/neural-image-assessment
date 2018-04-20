from path import Path
from PIL import Image as Image
import argparse
import os
from multiprocessing import Pool
from queue import Queue
import time
import progressbar


def resize_from_obj(img,dest,target_size=(224, 224),
    keep_aspect_ratio=True, resample='BILINEAR',jpg_quality=95):
    """
    resizes PIL image object img with resample as resampling filer then saves
    to new_path
    resample filters: Image.NEAREST (use nearest neighbour),
    Image.BILINEAR (linear interpolation),
    Image.BICUBIC (cubic spline interpolation), or
    Image.LANCZOS (a high-quality downsampling filter)
    """

    if resample == 'LANCZOS':
        resample = Image.LANCZOS
    elif resample == 'NEAREST':
        resample = Image.NEAREST
    elif resample == 'BILINEAR':
        resample = Image.BILINEAR
    elif resample =='BICUBIC':
        resample = Image.BICUBIC
    else:
        print("Invalid resample Function; using NEAREST")
        resample = Image.NEAREST

    if keep_aspect_ratio:
        img.thumbnail(target_size,resample)
    else:
        img = img.resize(target_size,resample)

    # img.save(dest)

    padding = Image.new('RGB',
                 target_size)
    padding.paste(img)
    padding.save(dest,'JPEG',quality=jpg_quality)

    #global _i
    #_i += 1

    # file_name = Path(img_path).name.lower()
    # print(file_name)
    return

def resize_from_path(img_path,dest,target_size=(224, 224),
    keep_aspect_ratio=True, resample='BILINEAR'):
    """
    resizes image at img_path with resample as resampling filer then saves
    to new_path
    resample filters: Image.NEAREST (use nearest neighbour),
    Image.BILINEAR (linear interpolation),
    Image.BICUBIC (cubic spline interpolation), or
    Image.LANCZOS (a high-quality downsampling filter)
    """
    img = Image.open(img_path)
    resize_from_obj(img,dest=dest,target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio, resample=resample)
    return


def split_list(list,parts=1):
    """split list in (nearly) equal sized parts"""
    p_len = len(list) / parts
    return [list[int(p_len*i):int(p_len*(i+1))] for i in range(parts)]

def _helper_resize_from_path(args):
    resize_from_path(*args)

def batch_resize(img_paths,dest_folder=None,target_size=(224, 224),
    keep_aspect_ratio=True, resample='BILINEAR', chunksize=16):
    """
    resizes all images given in list img_paths using multithreading. I advise using
    Pillow-SMID for further speedup.
    img_paths: list of paths to images
    dest_folder: images will be saved here.
    target_size: (int,int) size images will be resized to
    keep_aspect_ratio: if True will keep aspect ratio of each image and resized
                        s.t the long side fits target_size
    resample=Image.LANCZOS : resample filter used
    chunksize=16 : see multiprocessing Pool map_async
    """
    if dest_folder is None:
        dest_folder = img_paths + "/resize/"

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    total_imgs = len(img_paths)
    print("Resizing {} images... ".format(total_imgs), end='')
    #global _i
    #_i = 0
    #bar =  progressbar.ProgressBar(max_value=total_imgs)
    #bar.update(i)

    args = []
    for img_path in img_paths:
        args.append([img_path,dest_folder + Path(img_path).name.lower(),
        target_size, keep_aspect_ratio, resample])


    pool = Pool()
    pool.map(_helper_resize_from_path, args, chunksize)
    pool.close()

    #while not i >= total_imgs:
    #    bar.update(_i)
    #    time.sleep(1)

    pool.join()
    print("Done!")

def resize_folder(img_folder,dest_folder, include_subfolders=True,
    target_size=(224, 224), keep_aspect_ratio=True, resample='BILINEAR',
    chunksize=None):
    """
    resizes all images folder img_folder using multithreading. I advise using
    Pillow-SMID for further speedup.
    img_paths: list of paths to images
    dest_folder: images will be saved here.
    include_subfolders=True
    target_size: (int,int) size images will be resized to
    keep_aspect_ratio: if True will keep aspect ratio of each image and resized
                        s.t the long side fits target_size
    resample=Image.LANCZOS : resample filter used
    chunksize: if None it will use a heuristc that seems to work well,
                otherwise see multiprocessing Pool map_async,
    """
    print("test")
    if include_subfolders:
        img_paths = list(Path(img_folder).walkfiles('[!.]*.jpg'))
        img_paths += list(Path(img_folder).walkfiles('[!.]*.jpeg'))
        img_paths += list(Path(img_folder).walkfiles('[!.]*.png'))
    else:
        img_paths = list(Path(img_folder).files('[!.]*.jpg'))
        img_paths += list(Path(img_folder).files('[!.]*.jpeg'))
        img_paths += list(Path(img_folder).files('[!.]*.png'))

    if chunksize == None:
        # Heuristic for choosing the chunksize, seems to work well,
        # but most likely not optimal
        chunksize = max((1,min([len(img_paths)//32,128])))

    batch_resize(img_paths,dest_folder, target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio, resample=resample,
        chunksize=chunksize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize Images')
    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')
    parser.add_argument('-dest', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')
    parser.add_argument('-subf', type=str, default="true",
                        help='include images in subfolders')
    parser.add_argument('-aspect', type=str, default="true",
                        help='resized images will retain aspect ratio')
    args = parser.parse_args()
    target_size = (224, 224)
    include_subfolders = args.subf.lower() in ("true", "yes", "t", "1")
    keep_aspect_ratio = args.aspect.lower() in ("true", "yes", "t", "1")

    print(args.dest)
    if args.dir is not None:
        resize_folder(args.dir,args.dest, include_subfolders=include_subfolders,
        target_size=target_size, keep_aspect_ratio=keep_aspect_ratio,
            resample='LANCZOS', chunksize=None)
