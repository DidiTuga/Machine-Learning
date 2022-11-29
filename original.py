##
# Nome do aluno: Diogo Santos
#
#
##

import random

# cria instancias para o dataset
def create_instances(t):
    # ler ficheiro
    tempo = 20*t
    with open("csv/time_series_data_human_activities.csv", "r") as f:
        lines = f.readlines()
        # ir buscar as atividades e guardar as atividades num array
        atividades = []
        tamanho = len(lines)
        for i in range(1, tamanho):
            aux = lines[i].replace("\n", "")
            aux = aux.split(",")
            atividades.append(aux[1])
        atividades = set(atividades)
        # meter por ordem alfabetica
        atividades = sorted(atividades)
    # fazer as instancias
        instance = []
        for i in range(1, tamanho):
            linha = ""
            # tirar o \n
            linha_inicial = lines[i].replace("\n", "")
            # separar por virgulas
            linha_inicial = linha_inicial.split(",")
            # tirar o tempo
            id_i = linha_inicial[0]
            # index da atividade 0, 1, 2, 3, 4, 5 (como so tem 6 atividades)
            atividade_i = atividades.index(linha_inicial[1])
            # adicionar os 3 ultimos  a linha
            linha = ",".join(linha_inicial[-3:])
            # ler as t*20 linhas seguintes se existirem e adicionar ao fim da linha os 3 ultimos valores
            if i + tempo >= tamanho:
                break
            for j in range(i, (tempo)+i):
                if (tempo)+i < tamanho:
                    linha_final = lines[j].replace("\n", "")
                    linha_final = linha_final.split(",")
                    linha = linha + "," + ",".join(linha_final[-3:])
            # adicionar a linha ao instance
            linha = "".join(linha) + "," + str(atividade_i) + "," + id_i

            instance.append(linha)
        # escrever para o ficheiro
        with open("csv/instances.csv", "w") as f:
            for i in instance:
                f.write(i + "\n")

# cria os k fold sets e guarda num ficheiro 
def create_k_fold_sets(k):
    # criar k_fold_sets
    k_fold_sets = []
    with open("csv/instances.csv", "r") as f:
        lines = f.readlines()
        for i in range(k):
            k_fold_sets.append([])
        # adicionar as linhas aos k_fold_sets
        for i in range(len(lines)):
            k_fold_sets[i % k].append(lines[i])
        # juntar os k_fold_sets todos menos dois deles e guardar num ficheiro
    # Criar os k_folds
    for i in range(k):
        # ir buscar dois k_fold_sets random e meter num ficheiro teste e validacao e o restante num ficheiro treino
        k_fold_sets_aux = k_fold_sets.copy()
        fold_test = []
        pop_1 = random.randint(0, k-1)
        # ate k-2 porque o pop_1 ja foi escolhido
        pop_2 = random.randint(0, k-2)
        while pop_1 == pop_2:
            pop_2 = random.randint(0, k-2)
        # adicionar os dois k_fold_sets ao fold_test
        fold_test.append(k_fold_sets_aux.pop(pop_1))
        fold_test.append(k_fold_sets_aux.pop(pop_2))
        # juntar os k_fold_sets todos menos dois deles e guardar num ficheiro
        with open("csv/train/train_set_" + str(i) + ".csv", "w") as f:
            for j in k_fold_sets_aux:
                for n in j:
                    f.write(n)
        # guardar os dois k_fold_sets num ficheiro teste
        with open("csv/test/test_validation_set_" + str(i) + ".csv", "w") as f:
            for j in fold_test:
                for n in j:
                    f.write(n)


# 0 valor minimo da caracteristica
# 1 valor maximo da caracteristica
# x1 -> caracteristica 1
# y1 -> caracteristica 2
# z1 -> caracteristica 3
# novo valor da caracteristica = antigo valor da caracteristica - minimo da caracteristica / maximo da caracteristica - minimo da caracteristica
# para validar a validacao e o teste ele usa os valores do maximo e minimo do treino para normalizar os valores da validacao

def normalize(k):
    # ler ficheiro
    for n in range(k):
        with open("csv/train/train_set_"+str(n)+".csv", "r") as f:
            lines = f.readlines()
            # criar array com os valores maximos e minimos
            max_min = []
            tamanho = lines[0].replace("\n", "")
            tamanho = tamanho.split(",")
            tamanho = len(tamanho)-2
            # ir buscar os valores maximos e minimos
            for i in range(tamanho):
                max_min.append([float("inf"), float("-inf")])
            for i in range(len(lines)):
                linha = lines[i].replace("\n", "")
                linha = linha.split(",")
                for j in range(tamanho):
                    if float(linha[j]) > max_min[j][1]:
                        max_min[j][1] = float(linha[j])
                    if float(linha[j]) < max_min[j][0]:
                        max_min[j][0] = float(linha[j])
            # escrever o maximo na primeira linha e o minimo na segunda
            with open("csv/maxmin/max_min_"+str(n)+".csv", "w") as f:
                # escrever o maximo
                for i in max_min:
                    if i == (len(max_min)-1):
                        f.write(str(i[1]))
                    else:
                        f.write(str(i[1]) + ",")
                # escrever o minimo
                f.write("\n")
                for i in max_min:
                    if i == (len(max_min)-1):
                        f.write(str(i[0]))
                    else:
                        f.write(str(i[0]) + ",")
            # normalizar os valores do ficheiro de treino
        linhas = []
        with open("csv/train/train_set_"+str(n)+".csv", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                linha = lines[i].replace("\n", "")
                linha = linha.split(",")
                linha_str = ""
                for j in range(tamanho):
                    linha[j] = (float(linha[j]) - max_min[j][0]) / \
                        (max_min[j][1] - max_min[j][0])
                    if j == tamanho-1:
                        linha_str += str(linha[j]) + "," + \
                            linha[j+1] + "," + linha[j+2]
                    else:
                        linha_str += str(linha[j]) + ","
                linhas.append(linha_str)
                # escrever para o ficheiro
        with open("csv/train/train_set_"+str(n)+".csv", "w") as f:
            for i in linhas:
                f.write(i + "\n")
        # normalizar os valores do ficheiro de teste
        linhas = []
        with open("csv/test/test_validation_set_"+str(n)+".csv", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                linha = lines[i].replace("\n", "")
                linha = linha.split(",")
                linha_str = ""
                for j in range(tamanho):
                    linha[j] = (float(linha[j]) - max_min[j][0]) / \
                        (max_min[j][1] - max_min[j][0])
                    if j == tamanho-1:
                        linha_str += str(linha[j]) + "," + \
                            linha[j+1] + "," + linha[j+2]
                    else:
                        linha_str += str(linha[j]) + ","
                linhas.append(linha_str)
                # escrever para o ficheiro
        with open("csv/test/test_validation_set_"+str(n)+".csv", "w") as f:
            for i in linhas:
                f.write(i + "\n")


def main():
    # create_instances(1)
    k = 10
    #create_k_fold_sets(k)
    normalize(k)


main()
