package tensor

import "math/rand"

type Tensor struct {
	Data []float64
	Grad []float64
	Rows int
	Cols int
}

func New(rows, cols int) *Tensor {
	n := rows * cols
	return &Tensor{
		Data: make([]float64, n),
		Grad: make([]float64, n),
		Rows: rows,
		Cols: cols,
	}
}

func (t *Tensor) At(row, col int) float64 {
	return t.Data[row*t.Cols+col]
}

func (t *Tensor) Set(row, col int, val float64) {
	t.Data[row*t.Cols+col] = val
}
func (t *Tensor) AddAt(row, col int, val float64) {
	t.Data[row*t.Cols+col] += val
}
func (t *Tensor) GradAt(row, col int, val float64) {
	t.Grad[row*t.Cols+col] += val
}
func (t *Tensor) AddGrad(row, col int, val float64) {
	t.Grad[row*t.Cols+col] += val
}

func (t *Tensor) idx(row, col int) int {
	return row*t.Cols + col
}
func (t *Tensor) ZeroGrad() {
	for i := 0; i < t.Rows; i++ {
		t.Grad[i] = 0
	}
}

func (t *Tensor) RandNormal(mean, stddev float64) {
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64()*stddev + mean
	}
}

func (t *Tensor) Fill(v float64) {
	for i := range t.Data {
		t.Data[i] = v
	}
}
func (t *Tensor) Clone() *Tensor {
	c := New(t.Rows, t.Cols)
	copy(c.Data, t.Data)
	return c
}

func (t *Tensor) Row(i int) []float64 {
	return t.Data[i*t.Cols : (i+1)*t.Cols]
}
