#choose the basic image 
FROM ubuntu:20.04

#install python3
RUN apt-get update && \
    apt-get install -y python3 python3-pip

#run the command on the top of the alpine:3.8 basic image
RUN apt-get update && apt-get install --no-install-recommends -y \
    vim  \
    nano \ 
    && useradd -m -s /bin/bash yuzhen
    # echo "yuzhen ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

 
 
USER yuzhen 


# define the start working directory (the default directory is "/")
WORKDIR /home/yuzhen 


ENV app_port = 5000 \ 
    ABC = XYZ

COPY app.py /home/yuzhen/code/
 