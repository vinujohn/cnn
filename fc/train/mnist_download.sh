#!/usr/bin/env bash

trainingImages="train-images-idx3-ubyte"
trainingLabels="train-labels-idx1-ubyte"
testImages="t10k-images-idx3-ubyte"
testLabels="t10k-labels-idx1-ubyte"

trainImagesURL="https://raw.githubusercontent.com/HIPS/hypergrad/master/data/mnist/train-images-idx3-ubyte.gz"
trainLabelsURL="https://raw.githubusercontent.com/HIPS/hypergrad/master/data/mnist/train-labels-idx1-ubyte.gz"
testImagesURL="https://raw.githubusercontent.com/HIPS/hypergrad/master/data/mnist/t10k-images-idx3-ubyte.gz"
testLabelsURL="https://raw.githubusercontent.com/HIPS/hypergrad/master/data/mnist/t10k-labels-idx1-ubyte.gz"

outputDir="./fc/train/data" 

wget $trainImagesURL -O ${outputDir}/${trainingImages}.gz
wget $trainLabelsURL -O ${outputDir}/${trainingLabels}.gz
wget $testImagesURL -O ${outputDir}/${testImages}.gz
wget $testLabelsURL -O ${outputDir}/${testLabels}.gz

gzip -d ${outputDir}/${trainingImages}.gz
gzip -d ${outputDir}/${trainingLabels}.gz
gzip -d ${outputDir}/${testImages}.gz
gzip -d ${outputDir}/${testLabels}.gz