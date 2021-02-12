# EndoCV2021-polyp_det_seg_gen

**Maintained by:** [Sharib Ali](endocv.challenges@gmail.com)

## About:
This repo is designated for the evaluation of methods developed in [EndoCV2021](https://endocv2021.grand-challenge.org) (3rd International Endoscopy Computer Vision Challenge and Workshop in conjuenction to IEEE ISBI 2021) challenge, "Addressing generalisability in polyp detection and segmentation". Several tools that may help participants to assess their methods are provided. 

## Where to find what?

**Evaluating detection methods:**

- [Convert voc format to COCO format (to single json file)](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/voc2jsonCOCO.py)

- [Evaluate your prediction (COCO .json format)  output with GT (.json COCO format)](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/det_eval_coco.py)

- [Evaluate your prediction (VOC .txt format) with GT (.txt VOC format)](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/det_metrics_voc.py)
    *Deprecated!!! but participants can use if they wish*
    
**Evaluating generalisability in detection methods:**

- [Evaluation for testing generalisability between two detection results on multiple datasets](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/compute_det_gen.py)

**Evaluating segmentation methods:**

- [Evaluate segmentation methods for each test set separately](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/compute_seg.py)

**Evaluating generalisability in segmentation methods:**

- [Evaluate method generalisability](https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/blob/main/compute_seg_gen.py)

**References (details on generalisation tests can be found here, please cite these research if you use this code repo):**
[1] Ali, S., Zhou, F., Braden, B. et al. An objective comparison of detection and segmentation algorithms for artefacts in clinical endoscopy. Sci Rep 10, 2748 (2020). https://doi.org/10.1038/s41598-020-59413-5
