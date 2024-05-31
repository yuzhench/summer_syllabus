#check if there already exist a container called number_container
# container_index="$(docker ps -a -q -f name=number_countainer)"
# echo $container_index
# if [["$container_index"]]; then 
docker rm -f number_countainer
# fi

docker run -it \
    -v /home/yuzhen/Desktop/ml_learn/number_detection_project/model.py:/home/yuzhen/code/model.py \
    -v /home/yuzhen/Desktop/ml_learn/number_detection_project/mnist_number_detection.py:/home/yuzhen/code/mnist_number_detection.py \
    -v /home/yuzhen/Desktop/ml_learn/number_detection_project/test.py:/home/yuzhen/code/test.py \
    --name number_countainer \
    number_img 
