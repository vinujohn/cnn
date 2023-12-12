package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"

	"github.com/vinujohn/dnn-learning/fc"
)

const (
	ImagesMagicNum = 2051
	LabelsMagicNum = 2049

	LearningRate = 0.01
	Epochs       = 10
)

func main() {
	//train()
	test()
}

func train() {
	images := processImages("./fc/train/data/train-images-idx3-ubyte", ImagesMagicNum)
	labels, _ := processLabels("./fc/train/data/train-labels-idx1-ubyte", LabelsMagicNum)

	network := fc.NewFC(784, 800, 10)

	fmt.Println("***************************")
	fmt.Println("Begin Training")
	fmt.Printf("Learning Rate:%f Epochs:%d\n", LearningRate, Epochs)

	then := time.Now()
	network.Train(images, labels, LearningRate, Epochs)
	fmt.Println("Training Time:", time.Since(then))

	network.Save("LR_0_01_EP_10.gob")
}

func test() {
	images := processImages("./fc/train/data/t10k-images-idx3-ubyte", ImagesMagicNum)
	_, labels := processLabels("./fc/train/data/t10k-labels-idx1-ubyte", LabelsMagicNum)

	network, err := fc.Load("LR_0_01_EP_10.gob")
	if err != nil {
		panic(err)
	}

	fmt.Println("***************************")
	fmt.Println("Begin Testing")

	var yay int
	for i := range images {
		out := network.Predict(images[i])
		if labels[i] == indexOfMax(out) {
			yay++
		}
	}

	fmt.Printf("Num Correct:%d\n", yay)
	fmt.Printf("Percentage Correct:%d%%\n", yay/len(labels))
}

func indexOfMax(s []float64) byte {
	var max byte
	for i, v := range s {
		if v > s[max] {
			max = byte(i)
		}
	}
	return max
}

func processImages(filePath string, expectedMagicNum int) [][]float64 {
	f, err := os.Open(filePath)
	if err != nil {
		panic(fmt.Sprintf("could not open file. %v", err))
	}
	defer f.Close()

	header := make([]byte, 16)
	_, err = f.Read(header)
	if err != nil {
		panic(fmt.Sprintf("could not read header. %v", err))
	}

	magicNum := binary.BigEndian.Uint32(header[:4])
	if magicNum != uint32(expectedMagicNum) {
		panic(fmt.Sprintf("unexpected magic num. expecting %d, got %d", expectedMagicNum, magicNum))
	}

	numImages := binary.BigEndian.Uint32(header[4:8])
	rows := binary.BigEndian.Uint32(header[8:12])
	cols := binary.BigEndian.Uint32(header[12:16])
	numPixels := rows * cols

	fmt.Println("***************************")
	fmt.Println("processing images file", filePath)
	fmt.Printf("magicNum:%d, numImages:%d rows:%d cols:%d numPixels:%d\n", magicNum, numImages, rows, cols, numPixels)

	ret := make([][]float64, numImages)
	for i := 0; i < int(numImages); i++ {

		buf := make([]byte, numPixels)

		_, err = f.Read(buf)
		if err != nil {
			panic(fmt.Sprintf("could not read into buffer. %v", err))
		}

		ret[i] = convert(buf)
	}

	return ret
}

func processLabels(filePath string, expectedMagicNum int) ([][]float64, []byte) {
	f, err := os.Open(filePath)
	if err != nil {
		panic(fmt.Sprintf("could not open file. %v", err))
	}
	defer f.Close()

	header := make([]byte, 16)
	_, err = f.Read(header)
	if err != nil {
		panic(fmt.Sprintf("could not read header. %v", err))
	}

	magicNum := binary.BigEndian.Uint32(header[:4])
	if magicNum != uint32(expectedMagicNum) {
		panic(fmt.Sprintf("unexpected magic num. expecting %d, got %d", expectedMagicNum, magicNum))
	}

	numLabels := binary.BigEndian.Uint32(header[4:8])

	fmt.Println("***************************")
	fmt.Println("processing labels file", filePath)
	fmt.Printf("magicNum:%d, numLabels:%d\n", magicNum, numLabels)

	ret := make([][]float64, numLabels)

	buf := make([]byte, numLabels)

	_, err = f.Read(buf)
	if err != nil {
		panic(fmt.Sprintf("could not read into buffer. %v", err))
	}

	for i := 0; i < int(numLabels); i++ {
		ret[i] = make([]float64, 10) // 10 digits
		ret[i][buf[i]] = 1.0
	}

	return ret, buf
}

func convert(img []byte) []float64 {
	ret := make([]float64, len(img))

	for i := range img {
		ret[i] = float64(img[i])
	}

	return ret
}
