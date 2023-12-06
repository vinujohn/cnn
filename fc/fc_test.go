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
