import sys
import os
import numpy as np
import cv2
import math
import fileinput
import shutil

def increaseBbox(bbox, factor):
    tlx = bbox[0] 
    tly = bbox[1] 
    brx = bbox[2] 
    bry = bbox[3] 
    dx = factor
    dy = factor
    dw = 1 + factor
    dh = 1 + factor
    #Getting bbox height and width
    w = brx-tlx;
    h = bry-tly;
    tlx2 = tlx - w * dx
    tly2 = tly - h * dy
    brx2 = tlx + w * dw
    bry2 = tly + h * dh
    nbbox = np.zeros( (4,1), dtype=np.float32 )
    nbbox[0] = tlx2
    nbbox[1] = tly2
    nbbox[2] = brx2
    nbbox[3] = bry2 
    return nbbox

def image_bbox_processing_v2(img, bbox):
    img_h, img_w, img_c = img.shape
    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = bbox[2]
    rb_y = bbox[3]

    fillings = np.zeros( (4,1), dtype=np.int32)
    if lt_x < 0: ## 0 for python
        fillings[0] = math.ceil(-lt_x)
    if lt_y < 0:
        fillings[1] = math.ceil(-lt_y)
    if rb_x > img_w-1:
        fillings[2] = math.ceil(rb_x - img_w + 1)
    if rb_y > img_h-1:
        fillings[3] = math.ceil(rb_y - img_h + 1)
    new_bbox = np.zeros( (4,1), dtype=np.float32 )
    # img = [zeros(size(img,1),fillings(1),img_c), img]
    # img = [zeros(fillings(2), size(img,2),img_c); img]
    # img = [img, zeros(size(img,1), fillings(3),img_c)]

    # new_img = [img; zeros(fillings(4), size(img,2),img_c)]
    imgc = img.copy()
    if fillings[0] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [np.zeros( (img_h, fillings[0][0], img_c), dtype=np.uint8 ), imgc] )    
    if fillings[1] > 0:

        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [np.zeros( (fillings[1][0], img_w, img_c), dtype=np.uint8 ), imgc] )
    if fillings[2] > 0:


        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [ imgc, np.zeros( (img_h, fillings[2][0], img_c), dtype=np.uint8 ) ] )    
    if fillings[3] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [ imgc, np.zeros( (fillings[3][0], img_w, img_c), dtype=np.uint8) ] )


    new_bbox[0] = lt_x + fillings[0]
    new_bbox[1] = lt_y + fillings[1]
    new_bbox[2] = rb_x + fillings[0]
    new_bbox[3] = rb_y + fillings[1]
    return imgc, new_bbox

def preProcessImage(_savingDir, data_dict, data_root, factor, _alexNetSize, _listFile):
    #### Formatting the images as needed
    file_output = _listFile
    count = 1
    fileIn = open(file_output , 'w' )
    for key in  data_dict.keys():
        filename = data_dict[key]['file']
        im = cv2.imread(data_root +  filename)
        if im is not None:
            print 'Processing ' + filename + ' '+ str(count)
            sys.stdout.flush()
            lt_x = data_dict[key]['x']
            lt_y = data_dict[key]['y']
            rb_x = lt_x + data_dict[key]['width']
            rb_y = lt_y + data_dict[key]['height']
            w = data_dict[key]['width']
            h = data_dict[key]['height']
            center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
            side_length = max(w,h);
            bbox = np.zeros( (4,1), dtype=np.float32 )
            bbox[0] = center[0] - side_length/2
            bbox[1] = center[1] - side_length/2
            bbox[2] = center[0] + side_length/2
            bbox[3] = center[1] + side_length/2
            #img_2, bbox_green = image_bbox_processing_v2(im, bbox)
            #%% Get the expanded square bbox
            bbox_red = increaseBbox(bbox, factor)
            #[img, bbox_red] = image_bbox_processing_v2(img, bbox_red);
            img_3, bbox_new = image_bbox_processing_v2(im, bbox_red)
            #%% Crop and resized
            #bbox_red = ceil(bbox_red);
            bbox_new =  np.ceil( bbox_new )
            #side_length = max(bbox_new(3) - bbox_new(1), bbox_new(4) - bbox_new(2));
            side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
            bbox_new[2:4] = bbox_new[0:2] + side_length
            #crop_img = img(bbox_red(2):bbox_red(4), bbox_red(1):bbox_red(3), :);
            #resized_crop_img = imresize(crop_img, [227, 227]);# % re-scaling to 227 x 227
            bbox_new = bbox_new.astype(int)
            crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
            resized_crop_img = cv2.resize(crop_img, ( _alexNetSize, _alexNetSize ), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(_savingDir + key + '.jpg', resized_crop_img )
            
            

            ## Tracking pose image
            fileIn.write(key + ',')
            fileIn.write(_savingDir + key + '.jpg\n')
            
        else:
            print ' '.join(['Skipping image:', filename, 'Image is None', str(count)])
        count+=1
        #if count == 101:
        #    break
    fileIn.close()

def replaceInFile(filep, before, after):
    for line in fileinput.input(filep, inplace=True):
        print line.replace(before,after),

