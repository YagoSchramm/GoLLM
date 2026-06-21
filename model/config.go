package model

type Config struct {
	VocabSize  int
	ContextLen int
	EmbedDim   int
	NumHeads   int
	NumLayers  int
	FFNDim     int
}

func (t *Config) TinyConfig(vocabSize int) *Config {
	return &Config{
		VocabSize:  vocabSize,
		ContextLen: 64,
		EmbedDim:   64,
		NumHeads:   2,
		NumLayers:  2,
		FFNDim:     128,
	}
}

func (t *Config) SmallConfig(vocabSize int) *Config {
	return &Config{
		VocabSize:  vocabSize,
		ContextLen: 256,
		EmbedDim:   128,
		NumHeads:   4,
		NumLayers:  4,
		FFNDim:     512,
	}
}
