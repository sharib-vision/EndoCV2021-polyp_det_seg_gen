#!/bin/bash
#=========================================================================
#
# Project:   EAD2019 challenge
# Language:  bash
# Begin:     2019-03-05
#
# Author: Sharib Ali
# Responsible person: Sharib Ali <sharib.ali@eng.ox.ac.uk>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           FILES USED
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# evaluation_mAP-IoU/compute_mAP_IoU.py
# evaluation_semantic/semanticEval_dice_Jaccard_Overall.py
# overallEvaluations.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           FILE   STRUCTURE TO BE LOADED
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   - ead2019_testSubmission.zip
#       - detection_bbox
#       - semantic_masks
#       - generalization_bbox
# Please note: for semantic you will need to upload both semantic_bbox and semantic_masks (single folder is not accepted!)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BASE_DIR=/home/EndoCV2021/app
INPUT_FILES='/input'
MYDIR=$INPUT_FILES

# count number of directories
shopt -s nullglob
numfiles=($INPUT_FILES/*)
numfiles=${#numfiles[@]}

#MYDIR='/Users/shariba/development/deepLearning/EAD2020_codes/detection_bbox'
# BASE_DIR='/Users/shariba/development/deepLearning/EAD2020_codes'
# RESULT_FOLDER='./Result_test'
# `mkdir $RESULT_FOLDER`

DIRS=`ls -l $MYDIR | grep '^d' | awk '{print $9}'`
RESULT_FOLDER='/home/EndoCV2021/output'
#
#echo "only detection will be computed for this dataset, please make sure that you supply json file in COCO format...\n"
#echo "....make sure that all the directories of the test data are included..."

python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/det_eval_coco.py  --jsonPred $MYDIR/EndoCV2021/detection/EndoCV_DATA3.json --jsonGT $BASE_DIR/EndoCV2021_groundTruth-v1/detection/EndoCV_DATA3_GT.json --jsonResult $RESULT_FOLDER/metrics_det_EndoCV_DATA3.json
#
python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/det_eval_coco.py  --jsonPred $MYDIR/EndoCV2021/detection/EndoCV_DATA1.json --jsonGT $BASE_DIR/EndoCV2021_groundTruth-v1/detection/EndoCV_DATA1_GT.json --jsonResult $RESULT_FOLDER/metrics_det_EndoCV_DATA1.json
python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/det_eval_coco.py  --jsonPred $MYDIR/EndoCV2021/detection/EndoCV_DATA2.json --jsonGT $BASE_DIR/EndoCV2021_groundTruth-v1/detection/EndoCV_DATA2_GT.json --jsonResult $RESULT_FOLDER/metrics_det_EndoCV_DATA2.json
#
##      ---->  compute generalizability between 3 - 1 and 3 - 2
python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/compute_det_gen.py  --detectionMetric $RESULT_FOLDER/metrics_det_EndoCV_DATA3.json --generalizationMetric $RESULT_FOLDER/metrics_det_EndoCV_DATA1.json  --jsonFileName $RESULT_FOLDER/metric_gen_score_3_1.json
python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/compute_det_gen.py  --detectionMetric $RESULT_FOLDER/metrics_det_EndoCV_DATA3.json --generalizationMetric $RESULT_FOLDER/metrics_det_EndoCV_DATA2.json  --jsonFileName $RESULT_FOLDER/metric_gen_score_3_2.json
#
#
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##            COMPUTE THE FINAL METRICS.JSON for leaderboard
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#shopt -s nullglob
#numfiles=($RESULT_FOLDER/*json)
#numfiles=${#numfiles[@]}
#echo "COMPUTING FINAL METRICS....identified json files $numfiles"
RESULT_FOLDER_FINAL='/output'


echo "printing the final metric file...."
python $BASE_DIR/EndoCV2021-polyp_det_seg_gen/overallEvaluations_endocv2021_det.py \
--generalizationMetric_det_1 $RESULT_FOLDER/metric_gen_score_3_1.json \
--generalizationMetric_det_2 $RESULT_FOLDER/metric_gen_score_3_2.json \
--detectionMetric $RESULT_FOLDER/metrics_det_EndoCV_DATA3.json \
--caseType 1 \
--jsonFileName ${RESULT_FOLDER_FINAL}/metrics.json
#
