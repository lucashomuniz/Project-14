# ✅ PROJECT 14

Esse projeto vai aborar o processo de Machine Learning no contexto de um problema na área da Engenharia Civil. Porém, ao invés de aplicar tarefas uma a uma, criaremos módulos de automação. Ou seja, vamos desenvolver nosso próprio sistema de AutoML, sem o uso de framewords específicos e aplicando Machine Learning com o Spark MLlib no PySpark. O concreto é o material mais importante na Engenharia Civil. A resistência à compressão do concreto é uma função altamente não linear da idade e dos ingredientes utilizados. O trabalho será construir um modelo preditivo capaz de prever a resistência característica à compressão do concreto. Usaremos um dataframe disponível publicamente. A variável alvo será a "Concrete Compressive Strength" (coluna csMpa no dataframe) e as demais serão as variáveis preditodas. Como iremos prever um valor numérico e temos dados de entrada e saída, esté será o projeto de Regressão. Vamos experimentar diferentes algoritmos de regressão e escolher o que apresentar a melhor performance. Técnicas de otimização e hiperparâmetros serão exploradas para chegar ao melhor modelo possível. Com o modelo treinado faremos previsões usando novos dados.

Keywords: Python Language, Apache Spark, PySpark, Civil Engineering, Concrete Structures, Resistêencia à Compressão, Mpa, Machine Learning, Data Analysis, AutoML, Spark MLlib

# ✅ PROCESS

Com base na definição do problema de negócios e na análise do dataframe, geralmente é difícil determinar qual algoritmo é ideal para desenvolver o modelo de aprendizado de máquina. Portanto, o plano é iniciar um processo de experimentação, testando vários algoritmos com diferentes combinações de hiperparâmetros, criando assim diferentes versões dos modelos, e posteriormente comparando qual deles teve o melhor desempenho. Da mesma forma, antes de iniciar o projeto, é difícil saber quais são as técnicas de pré-processamento ideais. Portanto, faremos também um processo de experimentação nesse sentido.

Nesta fase, são realizadas tarefas como limpeza de dados (remoção de duplicatas e valores ausentes) e possíveis transformações específicas. O foco principal é entender o dataframe, visualizar os tipos de variáveis numéricas e categóricas, bem como suas distribuições e tratar outliers com base em boxplots, tabelas de descrição e contagens de frequência. Na Análise Exploratória, é importante não ter linhas duplicadas ou colunas duplicadas (variáveis), pois isso introduziria informações duplicadas e poderia enviesar o modelo desenvolvido. O objetivo é obter um modelo generalizável.

A próxima etapa é a Engenharia de Atributos, na qual são realizadas transformações mais profundas, se necessário, e variáveis podem ser criadas ou modificadas. Uma opção nesta etapa é a Seleção de Recursos, para selecionar as melhores variáveis para o processo de Aprendizado de Máquina. Além disso, uma técnica importante nessa etapa é a criação da Tabela de Correlação, que permite identificar possíveis níveis de relacionamento (positivo ou negativo) entre as variáveis, principalmente analisando evidências de multicolinearidade. A próxima etapa é o pré-processamento, onde são feitas alterações nas variáveis que ainda estão no formato de texto para convertê-las em números. 

Além disso, todo o modelo de Machine Learning é organizado, incluindo a escolha do algoritmo principal, codificação de rótulos, normalização, padronização e escalonamento de dados. Uma técnica amplamente utilizada nesta etapa é dividir o dataframe em conjuntos de treinamento e teste. Isso é importante porque o modelo de aprendizado de máquina é treinado nos dados de treinamento e, em seguida, avaliado nos dados de teste. Uma vez treinado o modelo, não é adequado apresentar os mesmos dados usados no treinamento, pois o modelo já os conhece. Para avaliar o desempenho do modelo, é necessário utilizar novos dados, cujos resultados já são conhecidos.

# ✅ CONCLUSION














# ✅ DATA SOURCES

https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
