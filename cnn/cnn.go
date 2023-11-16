// TODO: putting this on hold for FC work

package cnn

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	DefaultPadding = 0
	DefaultStride  = 1

	ErrInputDimNotEqual  = "input dimensions must be equal"
	ErrFilterDimNotEqual = "filter dimensions must be equal"
)

var (
	Padding uint
	Stride  uint
)

func Convolution(in, f *mat.Dense) (*mat.Dense, error) {
	inRowNum, inColNum := in.Dims()
	if inRowNum != inColNum {
		return nil, errors.New(ErrInputDimNotEqual)
	}

	filRowNum, filColNum := f.Dims()
	if filRowNum != filColNum {
		return nil, errors.New(ErrFilterDimNotEqual)
	}

	if Padding > 0 {
		in = padSides(in, uint(Padding))
	}

	if Stride <= 0 {
		Stride = DefaultStride
	}

	// (n + 2p - f) / s + 1
	newDim := ((inRowNum + (2 * int(Padding)) - filRowNum) / int(Stride)) + 1
	ret := mat.NewDense(newDim, newDim, nil)

	// perform convolution
	rows, cols := in.Dims()
	fmt.Printf("\n%v\n", mat.Formatted(in))
	for i := 0; i < rows-filRowNum+1; i += int(Stride) {
		for j := 0; j < cols-filColNum+1; j += int(Stride) {
			s := in.Slice(i, i+filRowNum, j, j+filColNum)
			fmt.Printf("\n%v\n", mat.Formatted(s))
			var temp *mat.Dense = mat.NewDense(filRowNum, filColNum, nil)
			temp.MulElem(s, f)
			data := temp.RawMatrix().Data
			var sum uint
			for _, v := range data {
				sum += uint(v)
			}
			//TODO: need to clean this up
			fmt.Println(sum)

		}
	}

	return ret, nil
}

func padSides(m *mat.Dense, amt uint) *mat.Dense {
	rows, cols := m.Dims()
	if amt == 0 {
		panic("invalid padding amount")
	}

	ret := mat.NewDense(rows+2*int(amt), cols+2*int(amt), nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			ret.Set(i+int(amt), j+int(amt), m.At(i, j))
		}
	}

	return ret
}
