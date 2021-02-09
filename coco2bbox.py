#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:06:53 2021

@author: shariba
@util
"""

import json
from misc import EndoCV_misc
from utils.bounding_box import BoundingBox
from utils.enumerators import BBFormat, BBType, CoordinatesType
from misc import EndoCV_misc

def coco2bbox(jsonFile, bb_type=BBType.GROUND_TRUTH):
    ret = []
    
    with open(jsonFile, "r") as f:
        json_object = json.load(f)
        
    classes = {}
    if 'categories' in json_object:
        classes = json_object['categories']
        # into dictionary
        classes = {c['id']: c['name'] for c in classes}
    images = {}
    # into dictionary
    for i in json_object['images']:
        images[i['id']] = {
            'file_name': i['file_name'],
            'img_size': (int(i['width']), int(i['height']))
        }
    annotations = []
    if 'annotations' in json_object:
        annotations = json_object['annotations']
        
    for annotation in annotations:
        img_id = annotation['image_id']
        x1, y1, bb_width, bb_height = annotation['bbox']
        if bb_type == BBType.DETECTED and 'score' not in annotation.keys():
            print('Warning: Confidence not found in the JSON file!')
            return ret
        confidence = annotation['score'] if bb_type == BBType.DETECTED else None
        # Make image name only the filename, without extension
        img_name = images[img_id]['file_name']
        img_name = EndoCV_misc.get_file_name_only(img_name)
        
        # create BoundingBox object
        bbox = BoundingBox(image_name=img_name,
                         class_id=classes[annotation['category_id']],
                         coordinates=(x1, y1, bb_width, bb_height),
                         type_coordinates=CoordinatesType.ABSOLUTE,
                         img_size=images[img_id]['img_size'],
                         confidence=confidence,
                         bb_type=1,
                         format=BBFormat.XYWH)
        
        ret.append(bbox)
    return ret