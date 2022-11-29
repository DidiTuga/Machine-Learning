##
# Nome do aluno: Diogo Santos
#
#
##
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
            atividade_i = atividades.index(linha_inicial[1]) # index da atividade 0, 1, 2, 3, 4, 5 (como so tem 6 atividades)
            # adicionar os 3 ultimos  a linha
            linha = ",".join(linha_inicial[-3:])  
            # ler as t*20 linhas seguintes se existirem e adicionar ao fim da linha os 3 ultimos valores
            if i + tempo >= tamanho:
                break
            for j in range(i, (tempo)+i):
                if (tempo)+i < tamanho:
                    linha_final = lines[j].replace("\n", "")
                    linha_final = linha_final.split(",")
                    linha =  linha + ",".join(linha_final[-3:])
            # adicionar a linha ao instance
            linha = "".join(linha) +  "," + str(atividade_i) + "," + id_i
            
            instance.append(linha)
        # escrever para o ficheiro
        with open("csv/instances.csv", "w") as f:
            for i in instance:
                f.write(i + "\n")

        
def create_k_fold_sets(k):
    # criar k_fold_sets
    with open("csv/instances.csv", "r") as f:
        lines = f.readlines()
        # criar k_fold_sets
        k_fold_sets = []
        for i in range(k):
            k_fold_sets.append([])
        # adicionar as linhas aos k_fold_sets
        for i in range(len(lines)):
            k_fold_sets[i%k].append(lines[i]) # PERGUNTAR SE QUEREM QUE SEJA RANDOM OU SEQUENCIAL
        # escrever para o ficheiro
        for i in range(k):
            with open("csv/k_fold_set_" + str(i) + ".csv", "w") as f:
                for j in k_fold_sets[i]:
                    f.write(j)
    
# TEMOS DE METER AINDA OS VALORES DO EXERCICIO POR ORDEM ALFABETICA E EM VEZ DE UMA STRING DAMOS UM ID AO EXERCICIO
    # ler ficheiro
    
def main():
    #create_instances(1)

    #create_k_fold_sets(10)


main()