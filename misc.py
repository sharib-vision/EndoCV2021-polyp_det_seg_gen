#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:51:51 2021

@author: sharib
"""
import operator
import sys
import numpy as np 
import os
from enum import Enum
import json

class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2
    
    
class EndoCV_misc:       
    def error(msg):
        print(msg)
        sys.exit(0)
        
    def is_float_between_0_and_1(value):
        """
        function to check if the number is a float between 0.0 and 1.0
        """
        try:
            val = float(value)
            if val > 0.0 and val < 1.0:
              return True
            else:
              return False
        except ValueError:
             return False
         
    def file_lines_to_list(path):
      with open(path) as f:
        content = f.readlines()
      content = [x.strip() for x in content]
      return content
  
    def get_file_name_only(file_path):
        if file_path is None:
            return ''
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def write2json(json_name, data):
        with open(json_name, 'w') as f:
            json.dump(data, f)
    