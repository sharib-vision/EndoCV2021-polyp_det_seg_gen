#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:57 2021

@author: sharib
"""

import os
import json
import glob

import cv2
import argparse
from misc import EndoCV_misc


coco_format = {
    "images": [
        {
        }
    ],
    "categories": [

    ],
    "annotations": [
        {
        }
    ]
}

def create_image_annotation(file_name, width, height, image_id):
    file_name = file_name.split('/')[-1] 
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images

def create_annotation_coco_format(min_x, min_y, width, height, score, image_id, category_id, annotation_id, args):
    bbox = (min_x, min_y, width, height)
    area = width * height

    if args.type == 'GT':
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
            'category_id': category_id,
            'segmentation': []
        }
    else:
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
            'category_id': category_id,
            'segmentation': [],
            'score': float(score)
        }

    return annotation

def images_annotations_info(args):
    
    root_path = args.root_path 
    dataset = {'categories': [], 'annotations': [], 'images': []}
     
    with open( 'obj.names') as f:
        classes = f.read().strip().split()
        
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
     
    global count, annot_count
    count = 0
    annot_count = 1
    
    bboxFolder = args.txtFiles_path
    ground_truth_files_list = glob.glob(os.path.join(bboxFolder,'*.txt'))
    ground_truth_files_list.sort()
   
    images = []
    annotations=[]
    score = []
    
    for txt_file in ground_truth_files_list:
        
        file_id = txt_file.split(".txt",1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = EndoCV_misc.file_lines_to_list(txt_file)
        
        # im = cv2.imread(os.path.join(root_path, 'images/') + file_id+'.jpg')
        im = cv2.imread(root_path + '/' +file_id+'.jpg')
        height, width, _ = im.shape
        images.append(create_image_annotation(os.path.join(root_path, '/') + file_id+'.jpg', width, height, count))
        
        for line in lines_list:
            try:
                annot_count+=1
                if args.type == 'GT':
                    cls_id, x1, y1, x2, y2 = line.split()
                else:
                    cls_id, score, x1, y1, x2, y2 = line.split()
   
                
                width_box = max(0, float(x2) - float(x1))
                height_box = max(0, float(y2)- float(y1))
                
            except ValueError:
                # error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                # EndoCV_misc.error(error_msg)
                if args.type == 'GT':
                    cls_id, x1, y1, x2, y2 = ('polyp', '-1', '-1', '-1', '-1')
                else: 
                    cls_id, x1, y1, x2, y2 = ('polyp', '-1', '-1', '-1', '-1')
                    score = 0.9
                
                width_box = max(0, float(x2) - float(x1))
                height_box = max(0, float(y2)- float(y1))
                
            annotations.append(create_annotation_coco_format(float(x1), float(y1), width_box, height_box, score, count, 1, annot_count, args))
            
        count = count+1
        
    return images, annotations
 
    
 
# def get_args():
#     parser = argparse.ArgumentParser('VOC format annotations to COCO dataset format')
#     parser.add_argument('--root_path', default='/media/sharib/development/EndoCV2021-test_analysis/endocv2021-test-noCopyAllowed-v1/EndoCV_DATA1', type=str, help='Absolute path for \'train.txt\' or \'test.txt\'')
#     parser.add_argument('--txtFiles_path', default='/media/sharib/development/EndoCV2021-test_analysis/codes-det/EndoCV2021/detection/EndoCV_DATA1_pred', type=str, help='Absolute')
#     parser.add_argument('--type', default='pred', type=str, help='Name the output json file')
#     args = parser.parse_args()
#     return args

def get_args():
    parser = argparse.ArgumentParser('VOC format annotations to COCO dataset format')
    parser.add_argument('--root_path', default='/Volumes/myPC/EndoCV2021-test_analysis/endocv2021-test-noCopyAllowed-v1/EndoCV_DATA2', type=str, help='Absolute path for \'train.txt\' or \'test.txt\'')
    parser.add_argument('--txtFiles_path', default='/Volumes/myPC/EndoCV2021-test_analysis/codes-det/EndoCV2021/detection/EndoCV_DATA2_pred', type=str, help='Absolute')
    parser.add_argument('--type', default='GT', type=str, help='Name the output json file')
    args = parser.parse_args()
    return args

# def get_args():
#     parser = argparse.ArgumentParser('Yolo format annotations to COCO dataset format')
#     parser.add_argument('--root_path', default='/Volumes/myPC/EndoCV2021/EndoCV2021-polyp_det_seg_gen/example/', type=str, help='Absolute path for \'train.txt\' or \'test.txt\'')
#     parser.add_argument('--txtFiles_path', default='/Volumes/myPC/EndoCV2021/EndoCV2021-polyp_det_seg_gen/example/predicted/', type=str, help='Absolute')
#     parser.add_argument('--type', default='Pred', type=str, help='Name the output json file')
#     args = parser.parse_args()
#     return args


if __name__ == '__main__':
    args = get_args()
    phase = 'EndoCV_DATA2_GT'
    classes = ['polyp']
 
    # folder = os.path.join(args.root_path, 'annotations')
    folder =  'annotations'
    if not os.path.exists(folder):
      os.makedirs(folder)
    
    coco_format['images'], coco_format['annotations'] = images_annotations_info(args)
    
    json_name = os.path.join('annotations/{}.json'.format(phase))
    
    for index, label in enumerate(classes):
        ann = {
            "supercategory": "none",
            "id": index + 1,  # Index starts with '1' .
            "name": label
        }
        coco_format['categories'].append(ann)
            
    with open(json_name, 'w') as f:
       json.dump(coco_format, f)
       
