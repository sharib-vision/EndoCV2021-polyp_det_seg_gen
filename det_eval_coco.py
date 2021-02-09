#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:37:47 2021

@author: shariba
: Change for EndoCV2021 challenge: 
    100 x 100: small 
    200 x 200: medium
    > 200 x 200
"""

import argparse
from misc import EndoCV_misc


def get_args():
    parser = argparse.ArgumentParser('Metrics for deteciton challenge of EndoCV2021 polyp (COCO json file format needed!!!)')
    parser.add_argument('--jsonGT', default='./examples/train_coco.json', type=str, help='Absolute path for \'train.txt\' or \'test.txt\'')
    parser.add_argument('--jsonPred', default='./examples/train_pred_coco.json', type=str, help='Absolute')
    parser.add_argument('--jsonResult', default='./examples/det_scores.json', type=str, help='Name the output json file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from coco2bbox import coco2bbox
    from utils.enumerators import  BBType
    from evaluationMetrics import coco_evaluator
    args = get_args()
    
    # default:
    gt_bbs =  coco2bbox(args.jsonGT)
    # for prediction
    pred_bbs =  coco2bbox(args.jsonPred, bb_type=BBType.DETECTED)
    #--------
    
    metric_values = coco_evaluator.get_coco_summary(gt_bbs, pred_bbs)
    
    print('Metric values for EndoCV2021 detection task:', metric_values)
    print('Also writing it to {}'.format(args.jsonResult))
    
    """
    Participants will be evaluated on AP (computed for IoU= .50:.05:95) and AP across scales (small, medium, large)
    However, different test datasets will be used for which an average values will be computed!!!
    """
    
    '----> save to json'
    EndoCV_misc.write2json(args.jsonResult, metric_values)
    
    
    
    
    
    