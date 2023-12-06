package fc

import (
	"math"
	"math/rand"
)

type layer struct {
	weights [][]float64
	bias    float64
	out     []float64
}

type FC struct {
	layers []*layer
	sizes  []uint
}

// "sizes" are the individual sizes of each layer in the network with
// the first layer being the input layer and the last layer being the
// output layer.  Minimum of 2 sizes are needed.
func NewFC(sizes ...uint) *FC {
	if len(sizes) < 2 {
		panic("sizes cannot be less than 2")
	}

	layers := make([]*layer, len(sizes)-1)

	for i, size := range sizes {
		if size < 1 {
			panic("invalid layer size. must be 1 or greater")
		}

		// skip the input layer
		if i == 0 {
			continue
		}

		// initialize weights and biases with random values
		layers[i-1] = &layer{
			weights: make([][]float64, size),
			bias:    rand.NormFloat64(),
			out:     make([]float64, size),
		}
		for j := range layers[i-1].weights {
			layers[i-1].weights[j] = make([]float64, sizes[i-1])
			for k := range layers[i-1].weights[j] {
				layers[i-1].weights[j][k] = rand.NormFloat64()
			}
		}
	}

	return &FC{
		layers: layers,
		sizes:  sizes,
	}
}

func (fc *FC) feedfoward(input []float64) {
	for _, layer := range fc.layers {
		for l, nodeWeights := range layer.weights {
			net := dot(nodeWeights, input) + layer.bias
			layer.out[l] = sigmoid(net)
		}
		input = layer.out
	}
}

func (fc *FC) backpropagation(truth []float64) {

}

func (fc *FC) updateWeights(learningRate float64) {

}

func (fc *FC) Predict(input []float64) []float64 {
	if len(input) != int(fc.sizes[0]) {
		panic("length of input does not equal expected input length")
	}

	fc.feedfoward(input)

	return fc.layers[len(fc.layers)-1].out
}

func (fc *FC) Train(dataset, truth [][]float64, learningRate float64, epochs uint) {
	if len(dataset) < 1 {
		panic("dataset must be populated for training")
	}
	if len(dataset) != len(truth) {
		panic("dataset length does not equal ground truth vector length")
	}
	if learningRate <= 0 {
		panic("learning rate must be a positive number")
	}
	if epochs <= 0 {
		panic("epochs must be a positive number")
	}

	for i := 0; i < int(epochs); i++ {
		for j, input := range dataset {
			fc.feedfoward(input)
			fc.backpropagation(truth[j])
			fc.updateWeights(learningRate)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1 - x)
}

func dot(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("vector inputs for dot product must be equal")
	}

	var sum float64
	for i := 0; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}
