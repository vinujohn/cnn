package fc

import "math/rand"

type layer struct {
	weights [][]float64
	bias    float64
	out     []float64
}

type FC struct {
	layers []*layer
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
	}
}
