package fc

import (
	"math"
	"math/rand"
)

type fcNode struct {
	Weights []float64 // weights for input from previous layer
	Delta   float64   // derivative of loss function * derivative of logistic function
	NetErr  []float64 // delta * derivative of input
	Output  float64   // output for this node
	Bias    float64   // bias for this node
}

type layer struct {
	Nodes []*fcNode
}

// FC is a fully connected network which is made up of layers of nodes.
// A minimum of 2 layers are needed with one being the input/hidden layer
// and the other being an output layer.  The input layer is not modeled
// in code but rather is just an input to the first hidden layer.
type FC struct {
	Layers []*layer
	Sizes  []uint // sizes of the individual layers. []sizes{input, hidden layers..., output}
}

// NewFC will create a network with the sizes specified with randomized
// weights and biases.  "sizes" are the individual sizes of each layer
// in the network with the first layer being the input layer and the last layer
// being the output layer.  Minimum of 2 sizes are needed.
func NewFC(sizes ...uint) *FC {
	if len(sizes) < 2 {
		panic("sizes cannot be less than 2")
	}

	layers := make([]*layer, len(sizes)-1)

	for i, size := range sizes[1:] {
		if size < 1 {
			panic("invalid layer size. must be 1 or greater")
		}

		layers[i] = &layer{
			Nodes: make([]*fcNode, size),
		}

		// initialize weights and biases with random values
		for j := range layers[i].Nodes {
			weights := make([]float64, sizes[i]) // len = previous layer node size
			for k := range weights {
				weights[k] = rand.NormFloat64()
			}

			layers[i].Nodes[j] = &fcNode{
				Weights: weights,
				Bias:    rand.NormFloat64(),
				NetErr:  make([]float64, sizes[i]), // len = previous layer node size
			}
		}
	}

	return &FC{
		Layers: layers,
		Sizes:  sizes,
	}
}

func (fc *FC) feedfoward(input []float64) {
	for _, layer := range fc.Layers {
		output := make([]float64, len(layer.Nodes))

		for i, node := range layer.Nodes {
			net := dot(node.Weights, input) + node.Bias
			node.Output = sigmoid(net)
			output[i] = node.Output
		}

		input = output
	}
}

func (fc *FC) backpropagation(input, truth []float64) float64 {
	if len(fc.output()) != len(truth) {
		panic("truth vector and output layer length do not match")
	}

	var loss float64

	// loop back through all layers
	for lIdx := len(fc.Layers) - 1; lIdx >= 0; lIdx-- {
		layer := fc.Layers[lIdx]
		// handle output layer
		if layer == fc.outputLayer() {
			for nIdx, node := range layer.Nodes {
				// derivative of loss function * derivative of logistic function
				node.Delta = (node.Output - truth[nIdx]) * sigmoidPrime(node.Output)

				// keep track of loss for stats
				loss += mse(truth[nIdx], node.Output)

				for wIdx := range node.Weights {
					// node.delta * derivative of net input function
					node.NetErr[wIdx] = node.Delta * fc.Layers[lIdx-1].Nodes[wIdx].Output
				}
			}
		} else {
			// handle input and hidden layers
			for nIdx, node := range layer.Nodes {
				for wIdx, _ := range node.Weights {
					for _, nextNode := range fc.Layers[lIdx+1].Nodes {
						node.Delta += nextNode.Delta * nextNode.Weights[nIdx]
					}

					node.Delta *= sigmoidPrime(node.Output)

					if lIdx == 0 {
						node.NetErr[wIdx] = node.Delta * input[wIdx]
					} else {
						node.NetErr[wIdx] = node.Delta * fc.Layers[lIdx-1].Nodes[wIdx].Output
					}
				}
			}
		}
	}

	return loss / float64(len(fc.outputLayer().Nodes))
}

func (fc *FC) updateWeightsAndBiases(learningRate float64) {
	for _, layer := range fc.Layers {
		for _, node := range layer.Nodes {
			for i := range node.Weights {
				node.Weights[i] -= learningRate * node.NetErr[i]
			}

			node.Bias -= learningRate * node.Delta
		}
	}
}

func (fc *FC) outputLayer() *layer {
	return fc.Layers[len(fc.Layers)-1]
}

func (fc *FC) output() []float64 {
	nodes := fc.outputLayer().Nodes
	output := make([]float64, len(nodes))

	for i := range output {
		output[i] = nodes[i].Output
	}

	return output
}

func (fc *FC) Predict(input []float64) []float64 {
	if len(input) != int(fc.Sizes[0]) {
		panic("length of input does not equal expected input length")
	}

	fc.feedfoward(input)

	return fc.output()
}

func (fc *FC) Train(dataset, truth [][]float64, learningRate float64) float64 {
	if len(dataset) < 1 {
		panic("dataset must be populated for training")
	}
	if len(dataset) != len(truth) {
		panic("dataset length does not equal ground truth vector length")
	}
	if learningRate <= 0 {
		panic("learning rate must be a number greater than zero")
	}

	loss := 0.0
	for j, input := range dataset {
		fc.feedfoward(input)

		loss += fc.backpropagation(input, truth[j])

		fc.updateWeightsAndBiases(learningRate)
	}

	return loss / float64(len(dataset))
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

func mse(target, output float64) float64 {
	return math.Pow(target-output, 2)
}
