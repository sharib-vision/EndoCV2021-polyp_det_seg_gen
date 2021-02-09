#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:50:14 2021

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
    valAppend_gen.append(data["EndoCV2021_det"]['AP'])
    valAppend_gen.append(data["EndoCV2021_det"]['APsmall'])
    valAppend_gen.append(data["EndoCV2021_det"]['APmedium'])
    valAppend_gen.append(data["EndoCV2021_det"]['APlarge'])
    return valAppend_gen


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="EndoCV2021: polyp generalization scoring", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--detectionMetric", type=str, default="./examples/test_det.json", help="json file for detection")
    parser.add_argument("--generalizationMetric", type=str, default="./examples/test_det.json", help="json file for generalization")
    parser.add_argument("--Result_dir", type=str, default="./examples/results", help="all evaluation scores used for grading")
    parser.add_argument("--jsonFileName", type=str, default="metric_gen_score.json", help="all evaluation scores used for grading")
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    import os
    import numpy as np
    from misc import EndoCV_misc 
    
    valArgs = get_args()
    exists_detect = os.path.isfile(valArgs.detectionMetric)
    exists_generalization = os.path.isfile(valArgs.generalizationMetric)

    jsonFilesDet = EndoCV_misc.get_all_det_eval_json('./examples')
    #TODO: sepearate by _det and use it for gen quantification, first remains the prime det!!!
    
    perCategoryDev = []
    arrayNames = ["polyp"]
    
    if exists_detect and exists_generalization:
        data_det = EndoCV_misc.read_json(valArgs.detectionMetric) 
        data_gen = EndoCV_misc.read_json(valArgs.generalizationMetric) 
        valAppend_det = []
        valAppend_gen = []
        
        valAppend_det = appendGen_measures(data_det)
        
        #@TODO: VDir - all det scores on generalisation test datasets
        valAppend_gen = appendGen_measures(data_gen)
        
        # tolerance limit of 10% is provided
        tol_limit = 10
        for i in range(0, len(valAppend_det)):
            deviation = computeDeviation(valAppend_det[i], valAppend_gen[i], tol_limit)
            perCategoryDev.append(deviation)
            
        # per class deviation
        meanDeviation = np.mean(perCategoryDev)
        
        # TODO: average over the number of test categories!!!
        mAP_g_test = valAppend_gen[0]
        
    else:
        print('generalization or detection file missing, both files are needed to compute this score')
        
        
    '''
    creating json file
    '''
    
    if exists_detect and exists_generalization:
        my_dictionary = {
            "EndoCV2021":{
                    "mAP_g":{
                     "value":(mAP_g_test) 
                    },
                    "score_g":{
                      "value": (meanDeviation),  
                    } 
                }
        }           
        # write to json      
        os.makedirs(valArgs.Result_dir, exist_ok=True)
        jsonFileName=os.path.join(valArgs.Result_dir, valArgs.jsonFileName)
        EndoCV_misc.write2json(jsonFileName, my_dictionary)
    
    