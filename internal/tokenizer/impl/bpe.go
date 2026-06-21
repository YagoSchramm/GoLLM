package impl

import (
	"sort"
	"strings"

	"github.com/YagoSchramm/GoLLM/internal/tokenizer"
)

func New() tokenizer.Tokenizer {
	return &BPE{}
}

type merges struct {
	a, b, result string
}

type BPE struct {
	vocab  map[string]int
	iVocab []string
	merges []merges
}

func (b *BPE) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(b.iVocab) {
			continue
		}
		token := b.iVocab[id]
		if strings.HasSuffix(token, "</w>") {
			sb.WriteString(strings.TrimSuffix(token, "</w>"))
			sb.WriteByte(' ')
		} else {
			sb.WriteString(token)
		}
	}
	return strings.TrimSpace(sb.String())
}

func (b *BPE) Encode(text string) []int {
	var ids []int
	for _, word := range strings.Fields(text) {
		chars := make([]string, 0, len(word)+1)
		for _, ch := range word {
			chars = append(chars, string(ch))
		}
		chars = append(chars, "</w>")
		for _, merge := range b.merges {
			chars = applyMergeSeq(chars, merge.a, merge.b, merge.result)
		}
		for _, token := range chars {
			if id, ok := b.vocab[token]; ok {
				ids = append(ids, id)
			}
		}
	}

	return ids
}

func (b *BPE) Train(corpus string, vocabSize int) {
	words := strings.Fields(corpus)
	seqs := make([][]string, len(words))
	for i, word := range words {
		chars := make([]string, 0, len(word)+1)
		for _, ch := range word {
			chars = append(chars, string(ch))
		}
		chars = append(chars, "</w>")
		seqs[i] = chars
	}

	b.vocab = make(map[string]int)
	b.iVocab = nil
	b.merges = nil
	for _, seq := range seqs {
		for _, token := range seq {
			if _, ok := b.vocab[token]; !ok {
				b.addToken(token)
			}
		}
	}

	for len(b.vocab) < vocabSize {
		counts := b.countPairs(seqs)
		if len(counts) == 0 {
			break
		}
		pair := b.bestPairs(counts)
		merged := pair[0] + pair[1]
		seqs = applyMergeAll(seqs, pair[0], pair[1], merged)
		b.merges = append(b.merges, merges{a: pair[0], b: pair[1], result: merged})
		b.addToken(merged)
	}
}

func (b *BPE) VocabSize() int {
	return len(b.vocab)
}

func (b *BPE) countPairs(seq [][]string) map[[2]string]int {
	counts := make(map[[2]string]int)
	for _, id := range seq {
		for i := 0; i+1 < len(id); i++ {
			counts[[2]string{id[i], id[i+1]}]++
		}
	}
	return counts
}

func (b *BPE) bestPairs(counts map[[2]string]int) [2]string {
	type entry struct {
		pair  [2]string
		count int
	}
	entries := make([]entry, 0, len(counts))

	for pair, count := range counts {
		entries = append(entries, entry{pair: pair, count: count})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].count != entries[j].count {
			return entries[i].count > entries[j].count
		}

		if entries[i].pair[0] != entries[j].pair[0] {
			return entries[i].pair[0] < entries[j].pair[0]
		}

		return entries[i].pair[1] < entries[j].pair[1]
	})

	return entries[0].pair
}

func applyMergeAll(seq [][]string, a, b, merged string) [][]string {
	out := make([][]string, len(seq))
	for i, tokens := range seq {
		out[i] = applyMergeSeq(tokens, a, b, merged)
	}

	return out
}

func applyMergeSeq(seq []string, a, b, merged string) []string {
	out := make([]string, 0, len(seq))
	i := 0
	for i < len(seq) {
		if i+1 < len(seq) && seq[i] == a && seq[i+1] == b {
			out = append(out, merged)
			i += 2
		} else {
			out = append(out, seq[i])
			i++
		}
	}

	return out
}

func (b *BPE) addToken(token string) {
	if _, ok := b.vocab[token]; ok {
		return
	}

	b.vocab[token] = len(b.iVocab)
	b.iVocab = append(b.iVocab, token)
}
