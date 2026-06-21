package model

import "github.com/YagoSchramm/GoLLM/internal/tensor"

type Embedding struct {
	Token    *tensor.Tensor
	Position *tensor.Tensor
	cache    struct{ ids []int }
}

func New(cfg Config) *Embedding {
	const std = 0.02
	tok := tensor.New(cfg.VocabSize, cfg.EmbedDim)
	tok.RandNormal(0, std)
	pos := tensor.New(cfg.ContextLen, cfg.EmbedDim)
	pos.RandNormal(0, std)
	return &Embedding{
		Token:    tok,
		Position: pos,
	}
}

func (e *Embedding) Forward(ids []int) *tensor.Tensor {
	T, D := len(ids), e.Token.Cols
	out := tensor.New(T, D)
	for t, id := range ids {
		for d := 0; d < D; d++ {
			out.Set(t, d, e.Token.At(id, d)+e.Position.At(t, d))
		}
	}
	e.cache.ids = ids
	return out
}

func (e *Embedding) Backward(grad *tensor.Tensor) *tensor.Tensor {
	for t, id := range e.cache.ids {
		for d := 0; d < grad.Cols; d++ {
			g := grad.At(t, d)
			e.Token.AddGrad(id, d, g)
			e.Position.AddGrad(t, d, g)

		}
	}
}
