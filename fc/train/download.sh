#!/usr/bin/env bash

trainingImages="train-images-idx3-ubyte"
trainingLabels="train-labels-idx1-ubyte"
testImages="t10k-images-idx3-ubyte"
testLabels="t10k-labels-idx1-ubyte"

baseURL1="https://web.archive.org/web/20220331225332/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
baseURL2="https://web.archive.org/web/20220331225243/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
baseURL3="https://web.archive.org/web/20220331225223if_/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
baseURL4="https://web.archive.org/web/20220331225222if_/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

outputDir="./fc/train/data" 

wget $baseURL -O ${outputDir}/${trainingImages}.gz ${baseURL1}
wget $baseURL -O ${outputDir}/${trainingLabels}.gz ${baseURL2}
wget $baseURL -O ${outputDir}/${testImages}.gz ${baseURL3}
wget $baseURL -O ${outputDir}/${testLabels}.gz ${baseURL4}

gzip -d ${outputDir}/${trainingImages}.gz
gzip -d ${outputDir}/${trainingLabels}.gz
gzip -d ${outputDir}/${testImages}.gz
gzip -d ${outputDir}/${testLabels}.gz