FROM continuumio/miniconda3
MAINTAINER Sharib Ali<sharib.ali@eng.ox.ac.uk> 

RUN conda create -n env python=3.7 numpy
RUN echo "source activate env" > ~/.bashrc

RUN mkdir /output/
RUN mkdir /input/

RUN chmod 777 /output/
RUN chmod 777 /input/

ENV PATH /opt/conda/envs/env/bin:$PATH

RUN conda install -c jmcmurray json 
RUN conda install -c conda-forge unzip 
RUN pip install tifffile
RUN pip install scikit-learn

RUN pip install --upgrade scikit-learn
RUN pip install numba==0.49.1

RUN pip install certifi==2019.11.28
RUN pip install cycler==0.10.0
RUN pip install kiwisolver==1.1.0
RUN pip install matplotlib==3.1.3
RUN pip install numpy==1.18.1
RUN pip install pyparsing==2.4.6
RUN pip install PyQt5==5.12.3
RUN pip install PyQt5-sip==4.19.18
RUN pip install PyQtWebEngine==5.12.1
RUN pip install python-dateutil==2.8.1
RUN pip install six==1.14.0
RUN pip install tornado==6.0.3
RUN pip install opencv-python==4.2.0.32

# create user ead2019
RUN useradd --create-home -s /bin/bash EndoCV2021
USER EndoCV2021

RUN mkdir -p /home/EndoCV2021/app 
WORKDIR /home/EndoCV2021/app

# add all evaluation and groundTruth directories
COPY EndoCV2021-polyp_det_seg_gen EndoCV2021-polyp_det_seg_gen/ 
COPY EndoCV2021_groundTruth-v1 EndoCV2021_groundTruth-v1/

# add run script

COPY run_script_det.sh run_script_det.sh

RUN [ "/bin/bash", "-c", "source activate env"]

RUN mkdir /home/EndoCV2021/input/
RUN mkdir /home/EndoCV2021/output/

# uncomment this for testing
#COPY ead2019_testSubmission /input/ead2019_testSubmission

#COPY EndoCV2021 /input/EndoCV2021

# COPY EndoCV2020_testSubmission/detection_bbox /input/detection_bbox
# COPY EndoCV2020_testSubmission/semantic_masks /input/semantic_masks
# COPY EndoCV2020_testSubmission/generalization_bbox /input/generalization_bbox


ENTRYPOINT /bin/bash

# ENTRYPOINT ["bash"]
# CMD ["/home/EndoCV2021/app/run_script_det.sh"]


# docker run --mount source=ead2019_testSubmission.zip,target=/input -ti --rm  ead2019_v2:latest /bin/bash
# sudo docker run -i -v /media/sharib/development/EndoCV2021-test_analysis/codes-det/test:/input -ti --rm  endocvleaderboard-det:latest /bin/bash
# --mount source=ead2019_testSubmission.zip,target=/input -ti --rm  endocv2020:latest /bin/bash
# sudo docker build -t endocvleaderboard-det:latest .
# sudo docker run -ti --rm endocvleaderboard-det:latest /bin/bash
# sudo docker save endocvleaderboard-det:latest | gzip -c > endocv2021.tar.gz
