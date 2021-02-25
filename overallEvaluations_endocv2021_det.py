#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:22:09 2019

@author: shariba

"""

import json

def appendGen_measures(data):
    valAppend_gen = []
    valAppend_gen.append(data["EndoCV2021_det"]['AP'])
    valAppend_gen.append(data["EndoCV2021_det"]['APsmall'])
    valAppend_gen.append(data["EndoCV2021_det"]['APmedium'])
    valAppend_gen.append(data["EndoCV2021_det"]['APlarge'])
    return valAppend_gen
       
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="For EAD2019 challenge: semantic segmentation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generalizationMetric_det_1", type=str, default="../Result_test/metrics_det_EAD2020.json", help="json file for detection")
    parser.add_argument("--generalizationMetric_det_2", type=str, default="../Result_test/metric_gen_score.json", help="json file for generalization")
    parser.add_argument("--detectionMetric", type=str, default="../Result_test/metrics_sem.json", help="son file for segmentation")
    parser.add_argument("--caseType", type=int, default=1, help="please set 0: only for dection both balanced, 1: only for instance segmentation only, 2: for generalization, 3: for all tasks")
    parser.add_argument("--Result_dir", type=str, default="finalEvaluationScores", help="all evaluation scores used for grading")
    parser.add_argument("--jsonFileName", type=str, default="metrics.json", help="all evaluation scores used for grading")
    args = parser.parse_args()
    return args

def read_json(jsonFile):
    with open(jsonFile) as json_data:
        data = json.load(json_data)
        return data
    
if __name__ == '__main__':
    import os
    import numpy as np
    
    valArgs = get_args()

    
    
    score_g_1 = 0
    score_g_2 = 0
    mAP_g_1 = 0
    # score d is based on mixed data distribution
    score_d = 0
    mAP_g_2 = 0
            
    debug = 1
    semScore_mean_dev=0

    valAppend_det=[]
    """ case: Semantic """
    if valArgs.caseType == 1:
        exists = os.path.isfile(valArgs.detectionMetric)
        if exists:
            data = read_json(valArgs.detectionMetric)

            valAppend_det = appendGen_measures(data)
            
            
            print(valAppend_det)
            
            # compute mean deviation
            score_d = np.mean(valAppend_det)
            
            
            if debug:
                print ('overall score for detection for EndoCV2021 challenge is:', score_d)
                print('~~~~~~~~~~~~~~~Complimentary informations~~~~~~~~~~~~~~~')
                print('number of semantic samples:', len(data))
                print('mean AP: {}, APsmall: {}, APmedium: {}, APlarge: {}:'.format(valAppend_det[0], valAppend_det[1], valAppend_det[2], valAppend_det[3]))
                print('~~~~~~~~~~~~~~~~~~~~~~E.O.F~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        

            if (valArgs.caseType == 1):
                ratioPass = 0
                
            """ case: Generalization """
        exists = os.path.isfile(valArgs.generalizationMetric_det_1)
        if exists:
            data = read_json(valArgs.generalizationMetric_det_1)
            valGen = []
            for p in data["EndoCV2021_det_gen"].values():
                valGen.append(p)
                
            if debug:
                print('~~~~~~~~~~~~~~~Complimentary informations~~~~~~~~~~~~~~~')
                print('mean seg_gen:', valGen[0]['value'])
                print('mean score_g:', valGen[1]['value'])
                print('~~~~~~~~~~~~~~~E.O.F~~~~~~~~~~~~~~~')
  
            mAP_g_1 = valGen[0]['value']
            score_g_1 = valGen[1]['value']
#    

        exists = os.path.isfile(valArgs.generalizationMetric_det_2)
        if exists:
            data = read_json(valArgs.generalizationMetric_det_2)
            valGen = []
            for p in data["EndoCV2021_det_gen"].values():
                valGen.append(p)
                
            if debug:
                print('~~~~~~~~~~~~~~~Complimentary informations~~~~~~~~~~~~~~~')
                print('mean seg_gen:', valGen[0]['value'])
                print('mean score_g:', valGen[1]['value'])
                print('~~~~~~~~~~~~~~~E.O.F~~~~~~~~~~~~~~~')
  
            mAP_g_2 = valGen[0]['value']
            score_g_2 = valGen[1]['value']
#    else:
#        print('no multi-class artefact detection found, mAPs are required for scoring both segmentation and generalization tasks')
#        
    '''
    creating json file
    '''
    # TODO: Loop this for 
    final_d_avg = (mAP_g_1+mAP_g_2+valAppend_det[0])/3
    my_dictionary = {
        "EndoCV2021":{
                "AP":{
                 "value":  (valAppend_det[0]) 
                },
                "APsmall":{
                 "value":   (valAppend_det[1]) 
                },
                "APmedium":{
                 "value":   (valAppend_det[2]) 
                },
                "APlarge":{
                  "value": (valAppend_det[3])
                },
                "score_d":{
                  "value": (score_d)
                },
                "mAP_g_1":{
                  "value": (mAP_g_1)
                },
                "mAP_g_2":{
                  "value": (mAP_g_2)
                },
                "dev_g_1":{
                  "value": (score_g_1)
                },
                "dev_g_2":{
                  "value": (score_g_2)
                },
                "final_dAvg":{
                  "value": (final_d_avg)
                }
            }
    }   
                
    # append json file             

    jsonFileName=valArgs.jsonFileName
    
    fileObj= open(jsonFileName, "a")
    # fileObj.write("\n")
    json.dump(my_dictionary, fileObj)
    fileObj.close()
    
            
            
        
        
