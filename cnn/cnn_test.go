package cnn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestStuff(t *testing.T) {
	m := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	s := m.Slice(1, 3, 1, 3)
	g := m.Grow(2, 2)

	t.Logf("m=\n%v", mat.Formatted(m))
	t.Logf("s=\n%v", mat.Formatted(s))
	t.Logf("g=\n%v", mat.Formatted(g))
}

func TestPadding(t *testing.T) {
	m := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	p := padSides(m, 1)
	t.Logf("m=\n%v", mat.Formatted(m))
	t.Logf("m=\n%v", mat.Formatted(p))
}

func TestConvolution(t *testing.T) {
	m := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	identity := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	Padding = 1
	Convolution(m, identity)
}
