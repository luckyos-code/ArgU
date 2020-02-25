#! /bin/bash

while getopts i:o: option
do
case "${option}"
in
i) inputDataset=${OPTARG};;
o) outputDir=${OPTARG};;
esac
done

docker build -t argu .
docker run --name argu-mongo -p 27017:27017 -d --rm mongo
docker run -v $inputDataset:/input -v $outputDir:/output --name argu --rm -it --network="host" argu
docker stop argu-mongo