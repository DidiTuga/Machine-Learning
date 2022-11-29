# Machine-Learning

### Trabalho realizado por Diogo Santos, aluno da Universidade da Beira Interior
### Professor Doutor Hugo Proença

O trabalho consiste em analizar uma sequencia de XYZ de um periodo e adivinhar o tipo de atividade que o utilizador está a fazer, isto através da analise de dados coletados de 36 users a praticar 6 atividades durante periodos de tempo.

## Steps

Step 1. Divide the available data into “Learning”, “Validation” and “Test” splits, using Kfold cross validation.


Step 2. Create your data instances, by concatenating “K” consecutive measurements from a users.


Step 3. Use a machine learning model to predict the users’ activity, based in the data created in Step 2.


Step 4. Obtain the corresponding ROC curves, for each class. Also, provide the accuracy value per class and the overall value.


Step 5. Find the importance of the X, Y and Z features to predict each activity.


Step 6. Repeat the experiments from Step 1-5, but using the subjects “ID” as the dependent variable. The hypothesis is: “Is it possible to identify a subject based on his activity measurements?”


Step 7. Both for the “Activity” and “ID” experiments, identify the instances in the test set that are particularly hard to classify. What characterizes such instances?
