#! /bin/bash

while getopts e:s:o: option
do
case "${option}"
in
e) embedding=${OPTARG};;
s) runType=${OPTARG};;
o) outputDir=${OPTARG};;
esac
done

docker run -e EMBEDDING=embedding -e RUN_TYPE=$runType -v $outputDir:/output --name argu --rm -it