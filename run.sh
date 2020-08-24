#! /bin/bash

while getopts n:e:s:o: option
do
case "${option}"
in
n) topN=${OPTARG};;
e) embedding=${OPTARG};;
s) runType=${OPTARG};;
o) outputDir=${OPTARG};;
esac
done

docker run -e TOP_N=$topN -e EMBEDDING=$embedding -e RUN_TYPE=$runType -v $outputDir:/output --name argu --rm -it argu