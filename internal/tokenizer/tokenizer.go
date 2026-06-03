package tokenizer

type Tokenizer interface {
	Train(corpus string, vocabSize int)
	Encode(text string) []int
	Decode(ids []int) string
	VocabSize() int
}
