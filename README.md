# Objetivo

Criar, através de uma técnica de aprendizado de máquina, um script que receba palavras em português e divida suas sílabas.

# Constatações
- Tamanho de input/output variável

# Considerar
- unichr

# Dúvidas
- Quais algoritmos podem ser mais eficientes que outros
- Pré-processamento está correto?
  1. Caso a palavra seja menor que a maior palavra em português, adicionar espaços até ficar do tamanho da maior (deixar de mesmo tamanho)
  2. Transformar sequência de caracteres em sequência de códigos unicode (numéricos)
  3. `from keras.utils import normalize`
