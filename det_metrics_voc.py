#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:12:16 2021

@author: endocv challenges
"""

import glob
import json
import os
import numpy as np 
from misc import EndoCV_misc, BBType

class EndoCV_det:
    def __init__(self,  nclass, minOverlap = 0.25):
        self.nclass = nclass
        self.minOverlap = minOverlap
        self.mAP = []

        
    def voc_ap(rec, prec):
      """
      --- Official matlab code VOC2012---
      """
      rec.insert(0, 0.0) 
      rec.append(1.0) 
      mrec = rec[:]
      prec.insert(0, 0.0)
      prec.append(0.0) 
      mpre = prec[:]
      """
       This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
      """
      for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
      """
       This part creates a list of indexes where the recall changes
      """
      i_list = []
      for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
          i_list.append(i) # if it was matlab would be i + 1
      """
       The Average Precision (AP) is the area under the curve
        (numerical integration)
      """
      ap = 0.0
      for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
      return ap, mrec, mpre
  
    
    def main_EndoCV2020(minOverlap, resultsfolder, gtfolder, predictfolder ):
        
        if minOverlap == []:
            minOverlap = EndoCV_det.minOverlap
            
        tmp_files_path = "tmp_files"
        if not os.path.exists(tmp_files_path): # if it doesn't exist already
          os.makedirs(tmp_files_path, exist_ok=True)
          
        results_files_path = resultsfolder
        os.makedirs(results_files_path, exist_ok=True)
        
        """
         Ground-Truth
           Load each of the ground-truth files into a temporary ".json" file.
           Create a list of all the class names present in the ground-truth (gt_classes).
        """
        # get a list with the ground-truth files
        ground_truth_files_list = glob.glob(os.path.join(gtfolder,'*.txt'))
        
        if len(ground_truth_files_list) == 0:
            EndoCV_misc.error("Error: No ground-truth files found!")
        ground_truth_files_list.sort()
        # dictionary with counter per class
        gt_counter_per_class = {}
        
        for txt_file in ground_truth_files_list:
          #print(txt_file)
          file_id = txt_file.split(".txt",1)[0]
          file_id = os.path.basename(os.path.normpath(file_id))
          # check if there is a correspondent predicted objects file
          if not os.path.exists(os.path.join(predictfolder , file_id + ".txt")):
            error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
        #    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            EndoCV_misc.error(error_msg)
          lines_list = EndoCV_misc.file_lines_to_list(txt_file)
          # create ground-truth dictionary
          bounding_boxes = []
          for line in lines_list:
            try:
              class_name, left, top, right, bottom = line.split()
            except ValueError:
              error_msg = "Error: File " + txt_file + " in the wrong format.\n"
              error_msg += " Expected: <class_name> <left> <top> <right> <bottom>\n"
              error_msg += " Received: " + line
              error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
              error_msg += "by running the script \"rename_class.py\" in the \"extra/\" folder."
              EndoCV_misc.error(error_msg)
            # check if class is in the ignore list, if yes skip
        
            bbox = left + " " + top + " " + right + " " +bottom
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            # count that object
            if class_name in gt_counter_per_class:
              gt_counter_per_class[class_name] += 1
            else:
              # if class didn't exist yet
              gt_counter_per_class[class_name] = 1
          # dump bounding_boxes into a ".json" file
          with open(os.path.join(tmp_files_path, file_id + "_ground_truth.json"), 'w') as outfile:
            json.dump(bounding_boxes, outfile)
        
        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        
        """
         Predicted
           Load each of the predicted files into a temporary ".json" file.
        """
        # get a list with the predicted files
        predicted_files_list = glob.glob(os.path.join(predictfolder, '*.txt'))
        predicted_files_list.sort()
        
        for class_index, class_name in enumerate(gt_classes):
          bounding_boxes = []
          for txt_file in predicted_files_list:
            #print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
              if not os.path.exists(os.path.join(gtfolder, file_id + ".txt")):
                error_msg = "Error. File not found: ground-truth/" +  file_id + ".txt\n"
        #        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                EndoCV_misc.error(error_msg)
            lines = EndoCV_misc.file_lines_to_list(txt_file)
            if lines==[]:
                  continue
            else:
                for line in lines:
                  try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                  except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    EndoCV_misc.error(error_msg)
                  if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                #print(bounding_boxes)
          # sort predictions by decreasing confidence
          bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
          with open(os.path.join(tmp_files_path, class_name + "_predictions.json"), 'w') as outfile:
            json.dump(bounding_boxes, outfile)
        
        """
         Calculate the AP for each class
        """
        sum_AP = 0.0
        ap_dictionary = {}
        sum_iou = 0.0
        iou_dictionary = {}
        
        # open file to store the results
        with open(os.path.join(results_files_path, "results.txt"), 'w') as results_file:
          results_file.write("# AP and precision/recall per class\n")
          count_true_positives = {}
          for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load predictions of that class
            """
            predictions_file = os.path.join(tmp_files_path, class_name + "_predictions.json")
            predictions_data = json.load(open(predictions_file))
        
            """
             Assign predictions to ground truth objects
            """
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            iou = [0] * nd
            for idx, prediction in enumerate(predictions_data):
              file_id = prediction["file_id"]
              
              gt_file = os.path.join(tmp_files_path, file_id + "_ground_truth.json")
              ground_truth_data = json.load(open(gt_file))
              ovmax = -1
              gt_match = -1
              # load prediction bounding-box
              bb = [ float(x) for x in prediction["bbox"].split() ]
              for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                  bbgt = [ float(x) for x in obj["bbox"].split() ]
                  bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                  iw = bi[2] - bi[0] + 1
                  ih = bi[3] - bi[1] + 1
                  if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                      ovmax = ov
                      gt_match = obj
        
              # set minimum overlap
              min_overlap = minOverlap
              
              if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                  # true positive
                  tp[idx] = 1
                  iou[idx] = ovmax
                  gt_match["used"] = True
                  count_true_positives[class_name] += 1
                  # update the ".json" file
                  with open(gt_file, 'w') as f:
                      f.write(json.dumps(ground_truth_data))
                else:
                  fp[idx] = 1
              else:
                fp[idx] = 1

            cumsum = 0
            for idx, val in enumerate(fp):
              fp[idx] += cumsum
              cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
              tp[idx] += cumsum
              cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
              rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
              prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        
            ap, mrec, mprec = EndoCV_det.voc_ap(rec, prec)
            sum_AP += ap
            
            ap_dictionary[class_name] = ap
            iou_dictionary[class_name] = np.mean(iou)
            sum_iou += np.mean(iou)
    
          mAP = sum_AP / n_classes
          mIoU = sum_iou / n_classes
          
        return mAP, mIoU, ap_dictionary