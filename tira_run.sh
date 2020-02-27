#! /bin/bash

while getopts s:i:o: option
do
case "${option}"
in
s) runType=${OPTARG};;
i) inputDataset=${OPTARG};;
o) outputDir=${OPTARG};;
esac
done

docker run --name argu-mongo -p 27017:27017 -d --rm mongo
docker run -e RUN_TYPE=$runType -v $inputDataset:/input -v $outputDir:/output --name argu --rm -it --network="host" argu
docker stop argu-mongo