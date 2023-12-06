package fc

import (
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
		assert.Len(t, fc.layers[0].weights, 2)
		assert.Len(t, fc.layers[0].weights[0], 3)
		assert.Len(t, fc.layers[0].weights[1], 3)

		// 1 node for layer 1 with 2 weights each
		assert.Len(t, fc.layers[1].weights, 1)
		assert.Len(t, fc.layers[1].weights[0], 2)
	})
}

// Example from: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example
func TestFeedFoward_Success(t *testing.T) {
	t.Run("2_2_2_network", func(t *testing.T) {
		fc := NewFC(2, 2, 2)
		// layer 0
		fc.layers[0].weights[0][0] = .15
		fc.layers[0].weights[0][1] = .20
		fc.layers[0].weights[1][0] = .25
		fc.layers[0].weights[1][1] = .30
		fc.layers[0].bias = .35

		// layer 1
		fc.layers[1].weights[0][0] = .40
		fc.layers[1].weights[0][1] = .45
		fc.layers[1].weights[1][0] = .50
		fc.layers[1].weights[1][1] = .55
		fc.layers[1].bias = .60

		fc.feedFoward([]float64{.05, .10})

		assert.InDelta(t, 0.75136507, fc.layers[1].out[0], 0.000000001)
	})
}
