#use the ubuntu 20.04 with the cuda as the base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
#use the python3
RUN apt-get update && apt-get install -y  python3 \
	python3-pip

#set the user
RUN useradd -m -s /bin/bash yuzhen
RUN apt-get update && apt-get install --no-install-recommends -y \
    vim  \
    nano \
	sudo 
#give the permission to the user 
RUN echo "yuzhen ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


#specify the user
USER yuzhen

#set the working space
WORKDIR /home/yuzhen

#copy the files to the container folders
#copy the code
# COPY /home/yuzhen/Desktop/ml_learn/number_detection_project/model.py /home/yuzhen/code/
# COPY /home/yuzhen/Desktop/ml_learn/number_detection_project/mnist_number_detection.py /home/yuzhen
# copy the requirnment
COPY  --chown=yuzhen:yuzhen requirements.txt /home/yuzhen/util/

# run the pip command to store the nessesary package
RUN pip install --user -r /home/yuzhen/util/requirements.txt

#environment
ENV MEANING=learn_how_to_write_dockerfile


#-v /home/yuzhen/Desktop/ml_learn/number_detection_project/model.py:/home/yuzhen/code/model.py


