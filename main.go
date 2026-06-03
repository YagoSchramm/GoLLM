package main

import (
	"fmt"

	"github.com/YagoSchramm/GoLLM/internal/tokenizer/impl"
)

func main() {
	token := impl.New()
	token.Train("O gato subiu no telhado o gato dormiu", 30)
	fmt.Println("Vocab Size:", token.VocabSize())
	ids := token.Encode("O gato subiu")
	fmt.Println(ids)
	fmt.Println(token.Decode(ids))

	ids2 := token.Encode("O cachorro correu pelo telhado")
	fmt.Println(ids2)
	fmt.Println(token.Decode(ids2))
}
