package fc

import (
	"math"
	"math/rand"
)

type fcNode struct {
	Weights []float64 // weights for input from previous layer
	Bias    float64   // bias for this node
	output  float64   // output for this node
	delta   float64   // derivative of loss function * derivative of activation function
	netErr  []float64 // delta * derivative of input
}

type layer struct {
	Nodes []*fcNode
}

// FC is a fully connected network which is made up of layers of nodes.
// A minimum of 2 layers are needed with one being the input/hidden layer
// and the other being an output layer.  The input layer is not modeled
// in code but rather is just an input to the first hidden layer.
type FC struct {
	Layers  []*layer
	Sizes   []uint    // sizes of the individual layers. []sizes{input, hidden layers..., output}
	outputs []float64 // output of the network after a feedfoward operation. activationFunc(net)
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
				weights[k] = uniformRandomWithoutZero()
			}

			layers[i].Nodes[j] = &fcNode{
				Weights: weights,
				Bias:    uniformRandomWithoutZero(),
				netErr:  make([]float64, sizes[i]), // len = previous layer node size
			}
		}
	}

	return &FC{
		Layers: layers,
		Sizes:  sizes,
	}
}

func (fc *FC) Predict(input []float64) []float64 {
	if len(input) != int(fc.Sizes[0]) {
		panic("length of input does not equal expected input length")
	}

	fc.feedfoward(input)

	return fc.outputs
}

func (fc *FC) Train(dataset, targets [][]float64, learningRate float64) float64 {
	if len(dataset) < 1 {
		panic("dataset must be populated for training")
	}
	if len(dataset) != len(targets) {
		panic("dataset length does not equal ground truth vector length")
	}
	if learningRate <= 0 {
		panic("learning rate must be a number greater than zero")
	}

	loss := 0.0
	for i, input := range dataset {
		fc.feedfoward(input)

		loss += fc.backpropagation(input, targets[i])

		fc.updateWeightsAndBiases(learningRate)
	}

	return loss / float64(len(dataset))
}

func (fc *FC) feedfoward(inputs []float64) {
	for _, layer := range fc.Layers {
		outputs := make([]float64, len(layer.Nodes))

		for i, node := range layer.Nodes {
			net := dot(node.Weights, inputs) + node.Bias

			// save the net output for nodes in the output layer
			// so we can use softmax on it later. use sigmoid on
			// all other layers.
			if layer == fc.outputLayer() {
				outputs[i] = net
			} else {
				node.output = sigmoid(net)
				outputs[i] = node.output
			}
		}

		if layer == fc.outputLayer() {
			softmax := softmaxVector(outputs)

			for i, node := range layer.Nodes {
				outputs[i] = softmax[i]
				node.output = outputs[i]
			}

			fc.outputs = outputs
		} else {
			inputs = outputs
		}
	}
}

func (fc *FC) backpropagation(input, targets []float64) float64 {
	if len(fc.outputs) != len(targets) {
		panic("truth vector and output layer length do not match")
	}

	// loop back through all layers
	for lIdx := len(fc.Layers) - 1; lIdx >= 0; lIdx-- {
		layer := fc.Layers[lIdx]
		// handle output layer
		if layer == fc.outputLayer() {
			for nIdx, node := range layer.Nodes {
				// derivative of loss function with softmax
				node.delta = (node.output - targets[nIdx])

				for wIdx := range node.Weights {
					// node.delta * derivative of net input function
					node.netErr[wIdx] = node.delta * fc.Layers[lIdx-1].Nodes[wIdx].output
				}
			}
		} else {
			// handle input and hidden layers
			for nIdx, node := range layer.Nodes {
				for wIdx, _ := range node.Weights {
					for _, nextNode := range fc.Layers[lIdx+1].Nodes {
						node.delta += nextNode.delta * nextNode.Weights[nIdx]
					}

					node.delta *= sigmoidPrime(node.output)

					if lIdx == 0 {
						node.netErr[wIdx] = node.delta * input[wIdx]
					} else {
						node.netErr[wIdx] = node.delta * fc.Layers[lIdx-1].Nodes[wIdx].output
					}
				}
			}
		}
	}

	return crossEntropy(targets, fc.outputs)
}

func (fc *FC) updateWeightsAndBiases(learningRate float64) {
	for _, layer := range fc.Layers {
		for _, node := range layer.Nodes {
			for i := range node.Weights {
				node.Weights[i] -= learningRate * node.netErr[i]
			}

			node.Bias -= learningRate * node.delta
		}
	}
}

func (fc *FC) outputLayer() *layer {
	return fc.Layers[len(fc.Layers)-1]
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1 - x)
}

func softmaxVector(zOutputs []float64) []float64 {
	var sum float64
	for _, v := range zOutputs {
		sum += math.Exp(v)
	}

	ret := make([]float64, len(zOutputs))
	for i := range zOutputs {
		ret[i] = math.Exp(zOutputs[i]) / sum
	}

	return ret
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

func crossEntropy(targets, outputs []float64) float64 {
	for i := range outputs {
		if targets[i] == 1.0 {
			return -math.Log(outputs[i])
		}
	}

	return 0
}

func uniformRandomWithoutZero() float64 {
	var rnd float64 = 0.0

	for rnd < 0.0099 {
		rnd = rand.Float64()
	}

	return rnd*2 - 1 // [-1, 1]
}
