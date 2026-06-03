package main

import (
	"fmt"

	"github.com/YagoSchramm/GoLLM/internal/tokenizer"
)

func main() {
	token := tokenizer.New()
	token.Train("O gato subiu no telhado o gato dormiu")
	fmt.Println("Vocab Size:", token.VocabSize())
	ids := token.Encode("O gato subiu")
	fmt.Println(ids)
	for _, id := range ids {
		fmt.Println(token.Decode(id))
	}

	ids2 := token.Encode("O cachorro correu pelo telhado")
	fmt.Println(ids2)
}
