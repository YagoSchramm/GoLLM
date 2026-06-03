package tokenizer

import "strings"

type Tokenizer struct {
	vocab   map[string]int
	reverse map[int]string
}

func New() *Tokenizer {
	return &Tokenizer{
		vocab:   make(map[string]int),
		reverse: make(map[int]string),
	}
}

func (t *Tokenizer) Train(corpus string) {
	for _, word := range strings.Fields(strings.ToLower(corpus)) {
		_, ok := t.vocab[word]
		if !ok {
			id := len(t.vocab)
			t.vocab[word] = id
			t.reverse[id] = word
		}
	}
}

func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, word := range strings.Fields(strings.ToLower(text)) {
		id, ok := t.vocab[word]
		if ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func (t *Tokenizer) Decode(id int) string {
	return t.reverse[id]
}

func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}
