import os
from typing import List
import random
import math

import numpy as np
from PIL import Image, ImageDraw

def display_boxes_from_img_and_list(img : Image, bboxs):
    '''
    img : PIL image
    bboxes : [[cls, x, y, w ,h]...] in yolo format
    '''

    for bbox in bboxs:
        _, x_cen, y_cen, bbox_width, bbox_height = bbox

        w,h = img.size
        tl = (int((x_cen-bbox_width/2)*w), int((y_cen-bbox_height/2)*h))
        br = (int((x_cen+bbox_width/2)*w), int((y_cen+bbox_height/2)*h))
        draw = ImageDraw.Draw(img)
        draw.rectangle([tl, br], outline='red')
    img.show()

def display_boxes_from_path(image_prefix_name):
    image_filename = image_prefix_name + '.jpg'
    loc_filename = image_prefix_name + '.txt'
    
    pil_img = Image.open(image_filename)
    bboxes = readAnnotation(loc_filename)
    display_boxes_from_img_and_list(pil_img, bboxes)

def readAnnotation(annotation_path) -> List[List[float]]:
    bboxes = []
    with open(annotation_path, 'r') as anno_file:
        lines = anno_file.readlines()
        for l in lines:
            l_strip = l.split()
            bbox = [int(l_strip[0]), 
                    float(l_strip[1]),
                    float(l_strip[2]), 
                    float(l_strip[3]),
                    float(l_strip[4])
                    ]
            bboxes.append(bbox)
    return bboxes

def findRelevantTrainingEx(original_training_list, name):
    '''Given a list of training images and text files, find
    the image and text file associated with a given trianing
    example name

    original_training_list: list[str] 
        containing AAAA.{jpeg, jpg etc.} and AAAA.txt
    name: str AAAA

    returns: Tuple of strings 
        (AAAA.{jpeg, jpg etc.}, AAAA.txt)
    '''
    all_rel = [s for s in original_training_list if os.path.splitext(s)[0] == name]
    if len(all_rel) == 0:
        raise KeyError(f'No such name {name} found in original training list')
    
    assert len(all_rel) == 2, 'Should only be 1 img and 1 txt file with matching name'

    if all_rel[0].endswith('txt'):
        return (all_rel[1], all_rel[0])
    else:
        return (all_rel[0], all_rel[1])

def findTargetDatasets(augmentation, local_storage_dir, local_datasets_list):
    apply_to = []
    if len(augmentation['target_datasets']) == 0:
        #Apply to all datasets
        apply_to = local_datasets_list
    else:
        #apply only to specified datasets
        dsets_tails = [os.path.split(s)[1] for s in local_datasets_list]

        #strip off .7z or .tar.xz if they are present
        #os.path.stripext wont get both .tar.xz
        targets_tails = [s[:s.find('.')] if s.find('.')  != -1 else s for s in augmentation['target_datasets']]

        for t in targets_tails:
            if t not in dsets_tails:
                print(f"===WARNING===\n\tAttempting to apply augmentation to dataset {t}\n" +
                        f"\t but it is not found in local datset list {local_datasets_list}")

        #collect
        apply_to = [os.path.join(local_storage_dir, d) for d in dsets_tails if d in targets_tails]
    return apply_to

#https://arxiv.org/pdf/1708.04896.pdf
def _selectRandomRectangleSubregion(width, height, s_l, s_h, r_1):
    max_attempts = 1000
    for _ in range(max_attempts):
        S_e = random.uniform(s_l, s_h) * width * height
        r_e = random.uniform(r_1, 1/r_1)

        H_e = int(math.sqrt(S_e*r_e))
        W_e = int(math.sqrt(S_e/r_e))

        x_e = random.randint(0,width)
        y_e = random.randint(0,height)
        if x_e + W_e <= width and y_e + H_e <= height:
            return (x_e, y_e), (x_e + W_e, y_e + H_e) 
    
    raise Exception("Couldn't find an internal rectangle")


def _addNoisyRectangle(pil_img, tl, br):
    noise_array_1c = np.random.randint(low=0, high = 255, size= (br[1]-tl[1], br[0]-tl[0]) )
    noise_array_3c = np.dstack([noise_array_1c]*3)
    img_array = np.array(pil_img)

    if img_array.squeeze().ndim == 2:
        img_array[tl[1]:br[1], tl[0]:br[0]] = noise_array_1c
    else:
        img_array[tl[1]:br[1], tl[0]:br[0],:] = noise_array_3c

    pil_img = Image.fromarray(img_array)
    return pil_img