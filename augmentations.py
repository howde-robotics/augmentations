import random

from PIL import Image, ImageEnhance

from augmentations.augment_util import readAnnotation, display_boxes_from_img_and_list
from augmentations.augment_util import _addNoisyRectangle, _selectRandomRectangleSubregion
# from augment_util import readAnnotation, display_boxes_from_img_and_list
# from augment_util import _addNoisyRectangle, _selectRandomRectangleSubregion



def change_brightness(img_path, annotation_path, factor):
    im = Image.open(img_path)

    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(im)

    pil_img = enhancer.enhance(factor)

    return pil_img, readAnnotation(annotation_path)

def augment_rotate_90(img_path, annotation_path):

    #Randomly rotate clockwise or counter clockwise
    clockwise = random.random() > 0.5

    im = Image.open(img_path)
    height_original = im.height
    width_original = im.width
    
    if clockwise:
        transposed_img  = im.transpose(Image.ROTATE_270)
    else:
        transposed_img  = im.transpose(Image.ROTATE_90)

    # width, height = im.size
    # width_T, height_T = transposed.size
    orig_bboxes = readAnnotation(annotation_path)
    new_bboxes = []
    for bbox in orig_bboxes:
        x1, y1, dx, dy = bbox[1:]

        p1 = ((x1-dx/2)*width_original,(y1-dy/2)*height_original)
        p2 = ((x1-dx/2)*width_original,(y1+dy/2)*height_original)
        p3 = ((x1+dx/2)*width_original,(y1-dy/2)*height_original)
        p4 = ((x1+dx/2)*width_original,(y1+dy/2)*height_original)

        height_T = width_original
        width_T = height_original
        
        if clockwise:
            p1_T = (width_T- p1[1], p1[0])
            p2_T = (width_T- p2[1], p2[0])
            p3_T = (width_T- p3[1], p3[0])
            p4_T = (width_T- p4[1], p4[0])
        else:
            p1_T = (p1[1], height_T-p1[0])
            p2_T = (p2[1], height_T-p2[0])
            p3_T = (p3[1], height_T-p3[0])
            p4_T = (p4[1], height_T-p4[0])

        new_bbox = [bbox[0],
                    (p1_T[0]+p2_T[0])/(2*height_original),
                    (p1_T[1]+p3_T[1])/(2*width_original),
                    dy,
                    dx
                    ]
        new_bboxes.append(new_bbox)
    return transposed_img, new_bboxes


#https://arxiv.org/pdf/1708.04896.pdf
def augment_random_erase(img_path, annotation_path, mode = 'image_object', s_l = 0.02, s_h = 0.4, r_1 = 0.3):
    pil_img = Image.open(img_path)
    orig_bboxes = readAnnotation(annotation_path)

    if 'image' in mode:
        tl, br = _selectRandomRectangleSubregion(pil_img.width, pil_img.height, s_l, s_h/4, r_1)
        pil_img = _addNoisyRectangle(pil_img, tl, br)
    
    if 'object' in mode:
        for bbox in orig_bboxes:
            _, x_cen_rel, y_cen_rel, w_rel, h_rel = bbox
            x_cen = int(x_cen_rel * pil_img.width)
            y_cen = int(y_cen_rel * pil_img.height)
            w_bbox = int(w_rel * pil_img.width)
            h_bbox = int(h_rel * pil_img.height)

            #don't need to random erase really tiny bboxes
            if w_bbox * h_bbox < 200:
                continue

            tl_rel, br_rel = _selectRandomRectangleSubregion(w_bbox, h_bbox, s_l, s_h, r_1)

            tl = (int(tl_rel[0] - w_bbox/2 + x_cen), int(tl_rel[1] - h_bbox/2 + y_cen)) 
            br = (int(br_rel[0] - w_bbox/2 + x_cen), int(br_rel[1] - h_bbox/2 + y_cen)) 

            pil_img = _addNoisyRectangle(pil_img, tl, br)
    
    return pil_img, orig_bboxes

def main():
    img_path = './big_ol_fake/train/random_erase_rotate_90_scene_1_01830.jpg'
    anno_path = './big_ol_fake/train/random_erase_rotate_90_scene_1_01830.txt'

    pil_img = Image.open(img_path)
    bboxes = readAnnotation(anno_path)
    display_boxes_from_img_and_list(pil_img, bboxes)

if __name__ == '__main__':
    main()