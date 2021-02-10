#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:39:29 2021

@author: shariba
"""

def computeDeviation(pDetect, pGener, tol_limit):
    """ per class deviation measurer"""
    expectedToleranceplus = pDetect+tol_limit*0.01*pDetect
    expectedToleranceminus = pDetect-tol_limit*0.01*pDetect
    
    if (pGener <= expectedToleranceplus and pGener >= expectedToleranceminus):
        deviation = 0
        print('within tolerance range, not penalized')
    else:
        deviation = abs(expectedToleranceminus-pGener)
    return deviation

def appendGen_measures(data):
    valAppend_gen = []
    valAppend_gen.append(data["EndoCV2021_seg"]['dice']['value'])
    valAppend_gen.append(data["EndoCV2021_seg"]['recall']['value'])
    valAppend_gen.append(data["EndoCV2021_seg"]['PPV']['value'])
    valAppend_gen.append(1-data["EndoCV2021_seg"]['hausdorff_distance']['value'])
    return valAppend_gen


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="EndoCV2021: polyp generalization scoring", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--segmentationMetric_test1", type=str, default="./examples/semantic_results/semantic_testType1.json", help="json file for detection")
    parser.add_argument("--segmentationMetric_test2", type=str, default="./examples/semantic_results/semantic_testType1.json", help="json file for generalization")
    parser.add_argument("--Result_dir", type=str, default="./examples/results", help="all evaluation scores used for grading")
    parser.add_argument("--jsonFileName", type=str, default="metric_seg_gen.json", help="all evaluation scores used for grading")
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    import os
    import numpy as np
    from misc import EndoCV_misc 
    
    valArgs = get_args()
    exists_seg = os.path.isfile(valArgs.segmentationMetric_test1)
    exists_gen = os.path.isfile(valArgs.segmentationMetric_test2)

    # jsonFilesDet = EndoCV_misc.get_all_det_eval_json('./examples')
    
    
    perCategoryDev = []
    arrayNames = ["polyp"]
    
    if exists_seg and exists_gen:
        data_det = EndoCV_misc.read_json(valArgs.segmentationMetric_test1) 
        data_gen = EndoCV_misc.read_json(valArgs.segmentationMetric_test2) 
        valAppend_seg = []
        valAppend_gen = []
        
        valAppend_seg = appendGen_measures(data_det)
        
        #@TODO: VDir - all det scores on generalisation test datasets
        valAppend_gen = appendGen_measures(data_gen)
        
        # tolerance limit of 10% is provided
        tol_limit = 10
        for i in range(0, len(valAppend_seg)):
            deviation = computeDeviation(valAppend_seg[i], valAppend_gen[i], tol_limit)
            perCategoryDev.append(deviation)
            
        # per class deviation
        meanDeviation = np.mean(perCategoryDev)
        
        # TODO: average over the number of test categories!!!
        
        #average the score over all
        seg_g_test = np.mean(valAppend_gen) 
        
    else:
        print('generalization or detection file missing, both files are needed to compute this score')
        
        
    '''
    creating json file
    '''
    
    if exists_seg and exists_gen:
        my_dictionary = {
            "EndoCV2021_seg_gen":{
                    "seg_g_score":{
                     "value":(seg_g_test) 
                    },
                    "score_seg_g":{
                      "value": (meanDeviation),  
                    } 
                }
        }           
        # write to json      
        os.makedirs(valArgs.Result_dir, exist_ok=True)
        jsonFileName=os.path.join(valArgs.Result_dir, valArgs.jsonFileName)
        EndoCV_misc.write2json(jsonFileName, my_dictionary)
    
    
    