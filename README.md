# Machine-Learning

### Trabalho realizado por Diogo Santos, aluno da Universidade da Beira Interior
### Professor Doutor Hugo Proença

O trabalho consiste em analizar uma sequencia de XYZ de um periodo e adivinhar o tipo de atividade que o utilizador está a fazer, isto através da analise de dados coletados de 36 users a praticar 6 atividades durante periodos de tempo.

## Resumo do código

Este é o código de um script em Python que lê um arquivo CSV com séries temporais de atividades humanas, cria instâncias a partir dessas séries temporais e, em seguida, divide essas instâncias em conjuntos de treino e teste para utilizar em uma rede neural de aprendizado supervisionado. As instâncias são criadas concatenando as séries temporais de atividades humanas com a atividade correspondente e um ID de usuário para cada atividade. O script também cria uma função para dividir essas instâncias em conjuntos de K folds e cria uma função para treinar uma rede neural MLP (Perceptron Multicamadas) usando esses conjuntos de folds. Além disso, o script possui uma função para calcular as curvas ROC (Receiver Operating Characteristic) e a matriz de confusão para avaliar o desempenho da rede neural.
