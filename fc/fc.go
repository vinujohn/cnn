package fc

import (
	"math"
	"math/rand"
)

type fcNode struct {
	weights []float64 // length = length of previous layer
	delta   float64   // derivative of loss function * derivative of logistic function
	netErr  []float64 // delta * derivative of input
	output  float64   // output for all nodes in this layer
}

type layer struct {
	nodes []*fcNode
	bias  float64 // same bias for all nodes in this layer
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

	for i, size := range sizes[1:] {
		if size < 1 {
			panic("invalid layer size. must be 1 or greater")
		}

		// initialize weights and biases with random values
		layers[i] = &layer{
			nodes: make([]*fcNode, size),
			bias:  rand.NormFloat64(),
		}
		for j := range layers[i].nodes {
			weights := make([]float64, sizes[i]) // length = previous layer node size
			for k := range weights {
				weights[k] = rand.NormFloat64()
			}

			layers[i].nodes[j] = &fcNode{
				weights: weights,
				netErr:  make([]float64, sizes[i]), // length = previous layer node size
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
		output := make([]float64, len(layer.nodes))
		for i, node := range layer.nodes {
			net := dot(node.weights, input) + layer.bias
			node.output = sigmoid(net)
			output[i] = node.output
		}
		input = output
	}
}

func (fc *FC) backpropagation(input, truth []float64) {
	if len(fc.output()) != len(truth) {
		panic("truth vector and output layer length do not match")
	}

	// loop back through all layers
	for i := len(fc.layers) - 1; i >= 0; i-- {
		layer := fc.layers[i]
		// handle output layer
		if layer == fc.outputLayer() {
			for j, node := range layer.nodes {
				for k := range node.weights {
					// derivative of loss function * derivative of logistic function
					node.delta = (node.output - truth[j]) * sigmoidPrime(node.output)
					// * derivative of net input function
					node.netErr[k] = node.delta * fc.layers[i-1].nodes[k].output
				}
			}
		} else {
			for j, node := range layer.nodes {
				for k := range node.weights {
					for _, nextNode := range fc.layers[i+1].nodes {
						node.delta += nextNode.delta * nextNode.weights[j]
					}
					if i == 0 {
						node.netErr[k] = node.delta * sigmoidPrime(node.output) * input[j]
					} else {
						node.netErr[k] = node.delta * sigmoidPrime(node.output) * node.weights[k]
					}
				}
			}
		}
	}
}

func (fc *FC) updateWeights(learningRate float64) {
	for _, layer := range fc.layers {
		for _, node := range layer.nodes {
			for i := range node.weights {
				node.weights[i] -= learningRate * node.netErr[i]
			}
		}
	}
}

func (fc *FC) outputLayer() *layer {
	return fc.layers[len(fc.layers)-1]
}

func (fc *FC) output() []float64 {
	nodes := fc.outputLayer().nodes
	output := make([]float64, len(nodes))

	for i := range output {
		output[i] = nodes[i].output
	}

	return output
}

func (fc *FC) Predict(input []float64) []float64 {
	if len(input) != int(fc.sizes[0]) {
		panic("length of input does not equal expected input length")
	}

	fc.feedfoward(input)

	return fc.output()
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
			fc.backpropagation(input, truth[j])
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
