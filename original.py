##
# Nome do aluno: Diogo Santos
#
#
##

import random
from numpy import array as np_array
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

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
            aux = 0
            for j in range((i+1), ((tempo)+i+1)):
                if (tempo)+i < tamanho:
                    linha_final = lines[j].replace("\n", "")
                    linha_final = linha_final.split(",")
                    if linha_final[0] == id_i and linha_final[1] == linha_inicial[1]:
                        aux = aux + 1
                        linha = linha + "," + ",".join(linha_final[-3:])
                    else:
                        break
                else:
                    break
            if (aux == tempo):
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
        # ver quantos ids existem
        ids = []
        for i in range(len(lines)):
            linha = lines[i].replace("\n", "")
            linha = linha.split(",")
            ids.append(linha[-1])
        ids = set(ids)
        ids = len(ids)
        # vai existir ids/k fold sets (sempre arredondado para baixo)
        id_fold_set = ids//k
        id_fold_set_rest = ids%k
        # percorrer o ficheiro e meter id_fold_set em cada fold set
        aux = -1
        ids_aux = []
        for i in range(ids):
            ids_aux.append([])
        for i in range(k):
            k_fold_sets.append([])
        ids_ = []
        for i in range(len(lines)):
            linha = lines[i].replace("\n", "")
            linha = linha.split(",")
            id_i = linha[-1]
            # se o id nao estiver no fold set e ainda nao tiver todos os ids volta do inicio
            if id_i not in ids_: 
                aux = aux + 1
                ids_.append(id_i)
            ids_aux[aux].append(lines[i])
        # meter os ids no fold set
        for i in range(k):
            for j in range(id_fold_set):
                k_fold_sets[i].append(ids_aux[i*id_fold_set+j])
        # meter os ids restantes no fold set
        for i in range(id_fold_set_rest):
            k_fold_sets[i].append(ids_aux[ids-id_fold_set_rest+i])
        # escrever nos treinos e testes
        # sendo dois fold sets para teste e o resto para treino
        fold_train = []
        fold_test = []
        for i in range(k):
            fold_train.append([])
            fold_test.append([])
            for j in range(k):
                if j == i or j == i+1:
                    fold_test[i].append(k_fold_sets[j])
                    if j == k-1:
                        fold_test[i].append(k_fold_sets[0])
                else:
                    fold_train[i].append(k_fold_sets[j])
        # escrever para o ficheiro
        for i in range(k):
            # juntar os k_fold_sets todos menos dois deles e guardar num ficheiro
            with open("csv/train/train_set_" + str(i) + ".csv", "w") as f:
                for j in fold_train[i]:
                    for n in j:
                        # list to string
                        n = "".join(n)
                        f.write(n)
            # guardar os dois k_fold_sets num ficheiro teste
            with open("csv/test/test_validation_set_" + str(i) + ".csv", "w") as f:
                for j in fold_test[i]:
                    for n in j:
                        n = "".join(n)
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

# Multi Layer Perceptron

def main():
    #create_instances(1)
    #k = 10
    #create_k_fold_sets(k)
    #normalize(k)
    # Multi Layer Perceptron
    
    NN = MLPClassifier((5,5))
    # Ler fichero de treino
    #for n in range(0,1):
    X = []
    y = []
    with open("csv/train/train_set_1.csv", "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            linha = lines[i].replace("\n", "")
            linha = linha.split(",")
            X.append(linha[:-2])
            y.append(linha[-2])
        y = np_array(y)
        label_encoder = LabelEncoder()
        inter_encoded = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False)
        inter_encoded = inter_encoded.reshape(len(inter_encoded), 1)
        y = onehot_encoder.fit_transform(inter_encoded)
        # Treinar o modelo
        NN.fit(X, y)
        # Ler ficheiro de teste
        print("Teste")
    yt = []
    with open("csv/test/test_validation_set_1.csv", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                linha = lines[i].replace("\n", "")
                linha = linha.split(",")
                X.append(linha[:-2])
                yt.append(linha[-2])
            yt = np_array(yt)
            label_encoder = LabelEncoder()
            inter_encoded = label_encoder.fit_transform(yt)
            onehot_encoder = OneHotEncoder(sparse=False)
            inter_encoded = inter_encoded.reshape(len(inter_encoded), 1)
            yt = onehot_encoder.fit_transform(inter_encoded)
            # Prever os valores
            # Testar o modelo
            y_pred = NN.predict(X)
            # Calcular a acuracia
            print("Acuracia: " + str(accuracy_score(yt, y_pred)))
            
# ROC Curve i (yt, y_pred) -> AUCi(Roc Curve i)
# media das AUCs -> AUCmédia
# desvio padrão das AUCs -> AUCdp
# AUCmédia +- AUCdp -> AUCmédia +- AUCdp
main()