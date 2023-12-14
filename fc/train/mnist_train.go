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

	LearningRate = 0.01
	Epochs       = 10

	ModelFile = "./fc/models/NN_200_80_10_LR_0_01_EP_10_%d.gob"
)

func main() {
	network := fc.NewFC(784, 200, 80, 10)

	then := time.Now()

	fmt.Printf("Starting Training/Test Cycle. Learning Rate:%f Epochs:%d\n", LearningRate, Epochs)

	lowestTrainingLoss, maxTestCorrect := math.MaxFloat64, 0
	for i := 1; i <= Epochs; i++ {
		trainImages := parseImages("./fc/train/data/train-images-idx3-ubyte", ImagesMagicNum)
		trainLabels, _ := parseLabels("./fc/train/data/train-labels-idx1-ubyte", LabelsMagicNum)

		fmt.Println("***************************")
		fmt.Println("Begin Training")

		loss := network.Train(trainImages, trainLabels, LearningRate)
		fmt.Printf("Epoch:%d Training Loss:%f Time So Far:%v\n", i, loss, time.Since(then))

		if loss < lowestTrainingLoss {
			lowestTrainingLoss = loss

			testImages := parseImages("./fc/train/data/t10k-images-idx3-ubyte", ImagesMagicNum)
			_, testLabels := parseLabels("./fc/train/data/t10k-labels-idx1-ubyte", LabelsMagicNum)

			fmt.Println("Begin Testing")

			var correct int
			for i := range testImages {
				out := network.Predict(testImages[i])
				if testLabels[i] == indexOfMax(out) {
					correct++
				}
			}

			percentageCorrect := float64(correct) / float64(len(testLabels)) * 100
			fmt.Printf("Epoch:%d Correct:%d %% Correct:%.2f%% Time So Far:%v\n", i, correct, percentageCorrect, time.Since(then))

			if correct > maxTestCorrect {
				maxTestCorrect = correct
				network.Save(fmt.Sprintf(ModelFile, i))
			} else {
				fmt.Println("correct number of tests is lower than the highest number of correct tests. exiting...")
				break
			}

			fmt.Println("***************************")

		} else {
			fmt.Println("training loss greater than lowest training loss. exiting...")
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

	fmt.Println("processing images file", filePath)
	fmt.Printf("magicNum:%d, numImages:%d rows:%d cols:%d numPixels:%d\n", magicNum, numImages, rows, cols, numPixels)

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

	fmt.Println("processing labels file", filePath)
	fmt.Printf("magicNum:%d, numLabels:%d\n", magicNum, numLabels)

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
		ret[i] = float64(img[i])
	}

	return ret
}
