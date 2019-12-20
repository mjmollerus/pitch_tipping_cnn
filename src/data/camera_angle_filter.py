import sys
import cv2
import numpy
import os

def match_images(template_dir,image_dir,threshold):
    '''
    Inputs: template directory path, image directory path, match threshold
    Outputs: list of strings of matched file names
    0.46 or there abouts seems to be a good threshold
    '''
    template_list = []
    
    template_file_list = os.listdir(template_dir)
    for item in template_file_list:
        template = cv2.imread(f'{template_dir}/{item}')
        template = template[:,:,2]
        template = template - cv2.erode(template, None)
        template_list.append(template)
    
    img_list = os.listdir(image_dir)[1:]
    match_list = []
    
    for file_name in img_list:
        img = cv2.imread(f'{image_dir}/{file_name}')
        img2 = img[:,:,2]
        img2 = img2 - cv2.erode(img2, None)
        
        ccnorm_tracker = 0
        for template in template_list:
            ccnorm = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
            if ccnorm.max() > ccnorm_tracker:
                ccnorm_tracker = ccnorm.max()
        if ccnorm_tracker >= threshold:
            match_list.append(file_name)
    
    return match_list

def clean_dir(image_dir,match_list):
    '''
    Inputs: image directory path, list of valid images from match_images()
    Outputs: Nothing
    Removes non-matching images from image directory
    '''
    
    
    image_list = os.listdir(image_dir)
    for file in image_list:
        if file not in match_list and os.path.exists(f'{image_dir}/{file}'):
            os.remove(f'{image_dir}/{file}')
            
    return

if __name__ == '__main__':
	match_list = match_images(template_dir,image_dir,threshold)
    clean_dir(image_dir,match_list)