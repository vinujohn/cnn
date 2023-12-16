package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/vinujohn/dnn-learning/fc"
)

// Explaination of MNIST dataset format
// https://web.archive.org/web/20220331230320/http://yann.lecun.com/exdb/mnist/

const (
	ImagesMagicNum = 2051
	LabelsMagicNum = 2049

	BaseURL           = "./fc/train/data/"
	TrainingImagesSet = BaseURL + "train-images-idx3-ubyte"
	TrainingLabelsSet = BaseURL + "train-labels-idx1-ubyte"
	TestingImagesSet  = BaseURL + "t10k-images-idx3-ubyte"
	TestingLabelsSet  = BaseURL + "t10k-labels-idx1-ubyte"

	LearningRate = 0.01
	Epochs       = 10

	ModelFile = "./fc/models/MNIST_200_80_10_LR_0_01_EP_10_%d.gob"
)

func main() {
	network := fc.NewFC(784, 200, 80, 10)

	then := time.Now()

	fmt.Printf("Starting Training/Test Cycle. Learning Rate:%f Epochs:%d\n", LearningRate, Epochs)
	fmt.Println("***************************")

	lowestTrainingLoss, maxTestCorrect := math.MaxFloat64, 0
	for i := 1; i <= Epochs; i++ {
		fmt.Println("Begin Training Epoch:", i)

		trainImages := parseImages(TrainingImagesSet, ImagesMagicNum)
		trainLabels, _ := parseLabels(TrainingLabelsSet, LabelsMagicNum)

		loss := network.Train(trainImages, trainLabels, LearningRate)
		fmt.Printf("Training Loss:%f Time:%s Time Elapsed:%v\n", loss, time.Now().Format("2006-01-02T15:04:05Z07:00"), time.Since(then))

		if loss < lowestTrainingLoss {
			lowestTrainingLoss = loss

			testImages := parseImages(TestingImagesSet, ImagesMagicNum)
			_, testLabels := parseLabels(TestingLabelsSet, LabelsMagicNum)

			fmt.Println("Begin Testing Epoch:", i)

			var correct int
			for i := range testImages {
				out := network.Predict(testImages[i])
				if testLabels[i] == indexOfMax(out) {
					correct++
				}
			}

			percentageCorrect := float64(correct) / float64(len(testLabels)) * 100
			fmt.Printf("Correct:%d %%Correct:%.2f%%\n", correct, percentageCorrect)

			if correct > maxTestCorrect {
				maxTestCorrect = correct
				err := network.Save(fmt.Sprintf(ModelFile, i))
				if err != nil {
					fmt.Printf("unable to save network model. %v\n", err)
				}
			} else {
				fmt.Println("expected number of correct too low. exiting...")
				break
			}

			fmt.Println("***************************")

		} else {
			fmt.Println("expected training loss too low. exiting...")
			break
		}
	}
}

func parseImages(filePath string, expectedMagicNum int) [][]float64 {
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

	//fmt.Println("processing images file", filePath)
	//fmt.Printf("magicNum:%d, numImages:%d rows:%d cols:%d numPixels:%d\n", magicNum, numImages, rows, cols, numPixels)

	ret := make([][]float64, numImages)
	for i := 0; i < int(numImages); i++ {

		buf := make([]byte, numPixels)

		numRead, err := f.Read(buf)
		if err != nil {
			panic(fmt.Sprintf("could not read into buffer. %v", err))
		}
		if numRead != int(numPixels) {
			panic(fmt.Sprintf("could not read correct number of pixels. %d vs %d", numRead, numPixels))
		}

		ret[i] = convertByteSliceToFloat64Slice(buf)
	}

	return ret
}

func parseLabels(filePath string, expectedMagicNum int) ([][]float64, []byte) {
	f, err := os.Open(filePath)
	if err != nil {
		panic(fmt.Sprintf("could not open file. %v", err))
	}
	defer f.Close()

	header := make([]byte, 8)
	_, err = f.Read(header)
	if err != nil {
		panic(fmt.Sprintf("could not read header. %v", err))
	}

	magicNum := binary.BigEndian.Uint32(header[:4])
	if magicNum != uint32(expectedMagicNum) {
		panic(fmt.Sprintf("unexpected magic num. expecting %d, got %d", expectedMagicNum, magicNum))
	}

	numLabels := binary.BigEndian.Uint32(header[4:8])

	//fmt.Println("processing labels file", filePath)
	//fmt.Printf("magicNum:%d, numLabels:%d\n", magicNum, numLabels)

	ret := make([][]float64, numLabels)

	buf := make([]byte, numLabels)

	numRead, err := f.Read(buf)
	if err != nil {
		panic(fmt.Sprintf("could not read into buffer. %v", err))
	}
	if numRead != int(numLabels) {
		panic(fmt.Sprintf("could not read correct number of labels. %d vs %d", numRead, numLabels))
	}

	for i := 0; i < int(numLabels); i++ {
		ret[i] = make([]float64, 10) // 10 digits
		ret[i][buf[i]] = 1.0
	}

	return ret, buf
}

// used during debugging
func writeFile(input []float64, index int, label, prediction byte) {

	// Open a file in append mode, create it if not exists, with write-only permissions
	file, err := os.OpenFile("./debug/file.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Write each byte as a two-digit hexadecimal number
	for i, b := range input {
		if i > 0 && i%28 == 0 {
			_, err := file.WriteString("\n")
			if err != nil {
				fmt.Println("Error writing to file:", err)
				return
			}
		}

		_, err := fmt.Fprintf(file, "%02x ", byte(b))
		if err != nil {
			fmt.Println("Error writing to file:", err)
			return
		}
	}

	// Add a new line at the end of the file
	_, err = file.WriteString("\n")
	if err != nil {
		fmt.Println("Error writing newline to file:", err)
		return
	}

	fmt.Fprintf(file, "index:%d label:%d prediction:%d\n", index, label, prediction)
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

func convertByteSliceToFloat64Slice(img []byte) []float64 {
	ret := make([]float64, len(img))

	for i := range img {
		ret[i] = float64(img[i]) / 255
	}

	return ret
}
