package fc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewFC_Panic(t *testing.T) {
	t.Run("invalid_num_layers", func(t *testing.T) {
		defer func(t *testing.T) {
			if r := recover(); r == nil {
				t.Fatal("expecting panic")
			}
		}(t)
		NewFC(1)
	})

	t.Run("invalid_layer_size", func(t *testing.T) {
		defer func(t *testing.T) {
			if r := recover(); r == nil {
				t.Fatal("expecting panic")
			}
		}(t)
		NewFC(1, 0)
	})
}

func TestNewFC_Success(t *testing.T) {
	t.Run("3_2_1_network", func(t *testing.T) {
		fc := NewFC(3, 2, 1)
		assert.Len(t, fc.layers, 2)

		// 2 nodes for layer 0 with 3 weights each
		assert.Len(t, fc.layers[0].nodes, 2)
		assert.Len(t, fc.layers[0].nodes[0].weights, 3)
		assert.Len(t, fc.layers[0].nodes[1].weights, 3)

		// 1 node for layer 1 with 2 weights each
		assert.Len(t, fc.layers[1].nodes, 1)
		assert.Len(t, fc.layers[1].nodes[0].weights, 2)
	})
}

func TestPredict_Success(t *testing.T) {
	t.Run("2_2_2_network", func(t *testing.T) {
		fc := Example_2_2_2_FC()

		prediction := fc.Predict([]float64{.05, .10})

		assert.InDelta(t, 0.75136507, prediction[0], 0.000000001)
		assert.InDelta(t, 0.772928465, prediction[1], 0.000000001)
	})
}

func TestTrain_Success(t *testing.T) {
	t.Run("2_2_2_network", func(t *testing.T) {
		fc := Example_2_2_2_FC()

		loss := fc.Train([][]float64{{.05, .10}}, [][]float64{{0.01, .99}}, 0.5, 10000)

		prediction := fc.Predict([]float64{.05, .10})

		t.Logf("Loss: %.20f", loss)

		assert.InDelta(t, 3e-6, loss, 1e-6)
		assert.InDelta(t, 0.01, prediction[0], 0.002)
		assert.InDelta(t, 0.99, prediction[1], 0.002)
	})
}

func nodesOutput(nodes []*fcNode) string {
	var s string

	for i, node := range nodes {
		s += fmt.Sprintf(`
		*****Node %d*****
		weights: %v
		delta: %v
		netErr: %v
		output: %v
		*****************
		`, i, node.weights, node.delta, node.netErr, node.output)
	}

	return s
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example
func Example_2_2_2_FC() *FC {
	fc := NewFC(2, 2, 2)
	// hidden layer
	fc.layers[0].nodes[0].weights[0] = .15
	fc.layers[0].nodes[0].weights[1] = .20
	fc.layers[0].nodes[1].weights[0] = .25
	fc.layers[0].nodes[1].weights[1] = .30
	fc.layers[0].bias = .35

	// output layer
	fc.layers[1].nodes[0].weights[0] = .40
	fc.layers[1].nodes[0].weights[1] = .45
	fc.layers[1].nodes[1].weights[0] = .50
	fc.layers[1].nodes[1].weights[1] = .55
	fc.layers[1].bias = .60

	return fc
}
