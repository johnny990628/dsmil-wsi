### TCGA
import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join 
import time, sys, warnings, glob
import threading
from tqdm import tqdm
import argparse
warnings.simplefilter('ignore')

def thres_saturation(img, t=15):
    # typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t

def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224): # position given as (y, x) at 5x scale
        img = img.read_region((position[0] * 4, position[1] * 4), 1, (patch_size, patch_size))
        img = np.array(img)[..., :3]
        if thres_saturation(img, 30):
            patch_name = "{}_{}".format(step[0], step[1])
            io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img))       
                        
def slide_to_patch(out_base, img_slides, step):
    makedirs(out_base, exist_ok=True)
    patch_size = 224
    step_size = step
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        dimension = img.level_dimensions[1] # given as width, height
        step_y_max = int(np.floor(dimension[1]/step_size)) # rows
        step_x_max = int(np.floor(dimension[0]/step_size)) # columns
        for j in range(step_y_max):
            for i in range(step_x_max):
                crop_slide(img, bag_path, (j*step_size, i*step_size), step=(j, i), patch_size=patch_size)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes patch extraction, attention computing, and stitching')
    path_base = ('../../../test/input')
    out_base = ('../../../test/patches')
    all_slides = glob.glob(join(path_base, '*.svs'))
    parser.add_argument('--thresholds_luad', type=float, default=0.5, help='Optimal threshold returned for LUAD')
    parser.add_argument('--thresholds_lusc', type=float, default=0.5, help='Optimal threshold returned for LUSC')
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=224)
    args = parser.parse_args()

    print('Cropping patches, please be patient')
    step = args.patch_size - args.overlap
    slide_to_patch(out_base, all_slides, step)