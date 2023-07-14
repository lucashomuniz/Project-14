"""
========
IMPORTS
========
"""

import pyspark
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Preparing the Spark Environment
# Creating the Spark Context
sc = SparkContext(appName = "Project-14")
sc.setLogLevel("ERROR")

# Creating the session
spark = SparkSession.builder.getOrCreate()

# Load the data
dados = spark.read.csv('dados/dataset.csv', inferSchema = True, header = True)
type(dados)

# Number of records
dados.count()

# Visualize the data in the default Spark DataFrame
dados.show(10)

# View data in Pandas format
dados.limit(10).toPandas()

# Schema
dados.printSchema()

"""
===========================================
DATA PREPARATION AUTOMATION MODULE
===========================================
"""

# MLlib requires all dataframe input columns to be vectorized.
# Let's create a Python function that will automate our data preparation work,
# including vectorization and all necessary tasks.
# First, let's list and remove missing values (if any).
# We will focus this project on Machine Learning, but always remember to check for missing values.

# We separate the missing data (if any) and remove it (if any)
dados_com_linhas_removidas = dados.na.drop()
print('Number of rows before removing missing values:', dados.count())
print('Number of rows after removing missing values:', dados_com_linhas_removidas.count())

# Data preparation function
def func_modulo_prep_dados(df, variaveis_entrada, variavel_saida, tratar_outliers = True, padronizar_dados = True):

    # Let's generate a new dataframe, renaming the argument that represents the output variable.
    novo_df = df.withColumnRenamed(variavel_saida, 'label')
    
    # We convert the target variable to numeric type as float (encoding)
    if str(novo_df.schema['label'].dataType) != 'IntegerType':
        novo_df = novo_df.withColumn("label", novo_df["label"].cast(FloatType()))
    
    # Checklists for variables
    variaveis_numericas = []
    variaveis_categoricas = []
    
    # If you have input variables of type string, convert to numeric type
    for coluna in variaveis_entrada:
        
        # Check if the variable is of type string
        if str(novo_df.schema[coluna].dataType) == 'StringType':
            
            # Define the variable with a suffix
            novo_nome_coluna = coluna + "_num"
            
            # Add to the list of categorical variables
            variaveis_categoricas.append(novo_nome_coluna)
            
        else:
            
            # If it is not a variable of type string, then it is numeric and we add it to the corresponding list
            variaveis_numericas.append(coluna)
            
            # We put the data in the dataframe of indexed variables
            df_indexed = novo_df

    # If the dataframe has data of type string, we apply indexing
    # Check that the list of categorical variables is not empty
    if len(variaveis_categoricas) != 0: 
        
        # Loop through the columns
        for coluna in novo_df:
            
            # If the variable is of type string, we create, train and apply the indexer
            if str(novo_df.schema[coluna].dataType) == 'StringType':
                
                # Create the indexer
                indexer = StringIndexer(inputCol = coluna, outputCol = coluna + "_num") 
                
                # Train and apply the indexer
                df_indexed = indexer.fit(novo_df).transform(novo_df)
    else:

        # If we don't have categorical variables anymore, then we put the data in the indexed variables dataframe
        df_indexed = novo_df
        
    # If it is necessary to handle outliers, we will do it now
    if tratar_outliers == True:
        print("\nApplying the treatment of outliers...")
        
        # Dictionary
        d = {}
        
        # Quartile dictionary of indexed dataframe variables (numeric variables only)
        for col in variaveis_numericas: 
            d[col] = df_indexed.approxQuantile(col,[0.01, 0.99], 0.25) 
        
        # Now we apply transformation depending on the distribution of each variable
        for col in variaveis_numericas:
            
            # We extract asymmetry from the data and use it to handle outliers
            skew = df_indexed.agg(skewness(df_indexed[col])).collect() 
            skew = skew[0][0]
            
            # We check for asymmetry and then apply:
            
            # Log transform + 1 if skewness is positive
            if skew > 1:
                indexed = df_indexed.withColumn(col, log(when(df[col] < d[col][0], d[col][0]).when(df_indexed[col] > d[col][1], d[col][1]).otherwise(df_indexed[col]) + 1).alias(col))
                print("\nA variável " + col + " was treated for positive asymmetry (right) with skew =", skew)
            
            # Exponential transformation if the skewness is negative
            elif skew < -1:
                indexed = df_indexed.withColumn(col,exp(when(df[col] < d[col][0], d[col][0]).when(df_indexed[col] > d[col][1], d[col][1]).otherwise(df_indexed[col])).alias(col))
                print("\nA variável " + col + " was treated for negative skewness (left) with skew =", skew)
                
            # Asymmetry between -1 and 1 we don't need to apply transformation to the data

    # Vectorization
    
    # Final list of attributes
    lista_atributos = variaveis_numericas + variaveis_categoricas

    # Create vectorizer for attributes
    vetorizador = VectorAssembler(inputCols = lista_atributos, outputCol = 'features')
    
    # Apply the vectorizer to the dataset
    dados_vetorizados = vetorizador.transform(df_indexed).select('features', 'label')
    
    # If the standardize_data flag is set to True, then we standardize the data by placing them on the same scale
    if padronizar_dados == True:
        print("\nStandardizing the dataset to the range 0 to 1...")
        
        # Create the scaler
        scaler = MinMaxScaler(inputCol = "features", outputCol = "scaledFeatures")

        # Compute the statistics summary and generate the standardizer
        global scalerModel
        scalerModel = scaler.fit(dados_vetorizados)

        # Defaults variables to range [min, max]
        dados_padronizados = scalerModel.transform(dados_vetorizados)
        
        # Generate the final data
        dados_finais = dados_padronizados.select('label', 'scaledFeatures')
        
        # Rename the columns (required by Spark)
        dados_finais = dados_finais.withColumnRenamed('scaledFeatures', 'features')
        
        print("\nProcess concluded!")

    # If the flag is set to False, then we don't standardize the data
    else:
        print("\nThe data will not be standardized because the standardizar_data flag has the value False.")
        dados_finais = dados_vetorizados
    
    return dados_finais

# Now we apply the data preparation module.

# List of input variables (all but the last one)
variaveis_entrada = dados.columns[:-1] 

# Target Variable
variavel_saida = dados.columns[-1] 

# Apply the function
dados_finais = func_modulo_prep_dados(dados, variaveis_entrada, variavel_saida)

# View
dados_finais.show(10, truncate = False)

"""
=======================
VERIFYING CORRELATION
=======================
"""

# Let's make sure we don't have multicollinearity before we move on.
# Remember the following guidelines for the Pearson Correlation Coefficient:
# - .00-.19 (very weak correlation)
# - .20-.39 (weak correlation)
# - .40-.59 (moderate correlation)
# - .60-.79 (strong correlation)
# - .80-1.0 (very strong correlation)

# Extract the correlation
coeficientes_corr = Correlation.corr(dados_finais, 'features', 'pearson').collect()[0][0]

# Convert the result to an array
array_corr = coeficientes_corr.toArray()

# List the correlation between the attributes and the target variable
for item in array_corr:
    print(item[7])

# Split into Training and Test Data
# Division with 70/30 ratio
dados_treino, dados_teste = dados_finais.randomSplit([0.7,0.3])

"""
===========================================
MÓDULO AUTOML (AUTOMATED MACHINE LEARNING)
===========================================
"""

# Let's create a function to automate the use of several algorithms.
# Our function will create, train and evaluate each of them with different combinations of hyperparameters.
# And then we'll choose the best performing model.

# Machine Learning module
def func_modulo_ml(algoritmo_regressao):
    # Function to get the type of the regression algorithm and create the object instance
    # We will use this to automate our process
    def func_tipo_algo(algo_regressao):
        algoritmo = algo_regressao
        tipo_algo = type(algoritmo).__name__
        return tipo_algo
    
    # Apply the previous function
    tipo_algo = func_tipo_algo(algoritmo_regressao)

    # If the algorithm is Linear Regression, enter this block if
    if tipo_algo == "LinearRegression":
        
        # We train the first version of the model without cross-validation
        modelo = regressor.fit(dados_treino)
        
        # Model metrics
        print('\033[1m' + "Linear Regression Model Without Cross Validation:" + '\033[0m')
        print("")
        
        # Evaluate the model with test data
        resultado_teste = modelo.evaluate(dados_teste)

        # Print model error metrics with test data
        print("RMSE in Test: {}".format(resultado_teste.rootMeanSquaredError))
        print("R2 Coefficient in Test: {}".format(resultado_teste.r2))
        print("")
        
        # Now let's create the second version of the model with the same algorithm, but using cross-validation
        
        # Prepare the hyperparameter grid
        paramGrid = (ParamGridBuilder().addGrid(regressor.regParam, [0.1, 0.01]).build())
        
        # Create the evaluators
        eval_rmse = RegressionEvaluator(metricName = "rmse")
        eval_r2 = RegressionEvaluator(metricName = "r2")

        # Create the Cross Validator
        crossval = CrossValidator(estimator = regressor,
                                  estimatorParamMaps = paramGrid,
                                  evaluator = eval_rmse,
                                  numFolds = 3) 
        
        print('\033[1m' + "Linear Regression Model With Cross Validation:" + '\033[0m')
        print("")
        
        # Train the model with cross validation
        modelo = crossval.fit(dados_treino)
        
        # Save the best version 2 model
        global LR_BestModel 
        LR_BestModel = modelo.bestModel
                
        # Predictions with test data
        previsoes = LR_BestModel.transform(dados_teste)
        
        # Evaluation of the best model
        resultado_teste_rmse = eval_rmse.evaluate(previsoes)
        print('RMSE in Test:', resultado_teste_rmse)
        
        resultado_teste_r2 = eval_r2.evaluate(previsoes)
        print('R2 Coefficient in Test:', resultado_teste_r2)
        print("")
    
        # List of columns to put in the summary dataframe
        columns = ['Regressor', 'Resultado_RMSE', 'Resultado_R2']
        
        # Format the results and create the dataframe
        
        # Format metrics and algorithm name
        rmse_str = [str(resultado_teste_rmse)] 
        r2_str = [str(resultado_teste_r2)] 
        tipo_algo = [tipo_algo] 
        
        # create dataframe
        df_resultado = spark.createDataFrame(zip(tipo_algo, rmse_str, r2_str), schema = columns)
        
        # Write the results to the dataframe
        df_resultado = df_resultado.withColumn('Result_RMSE', df_resultado.Resultado_RMSE.substr(0, 5))
        df_resultado = df_resultado.withColumn('Result_R2', df_resultado.Resultado_R2.substr(0, 5))
        
        return df_resultado

    else:
        
        # Check if the algorithm is the Decision Tree and create the hyperparameter grid
        if tipo_algo in("DecisionTreeRegressor"):
            paramGrid = (ParamGridBuilder().addGrid(regressor.maxBins, [10, 20, 40]).build())

        # Check if the algorithm is Random Forest and create the hyperparameter grid
        if tipo_algo in("RandomForestRegressor"):
            paramGrid = (ParamGridBuilder().addGrid(regressor.numTrees, [5, 20]).build())

        # Check if the algorithm is GBT and create the hyperparameter grid
        if tipo_algo in("GBTRegressor"):
            paramGrid = (ParamGridBuilder().addGrid(regressor.maxBins, [10, 20]).addGrid(regressor.maxIter, [10, 15]).build())
            
        # Check if the algorithm is Isotonic
        if tipo_algo in("IsotonicRegression"):
            paramGrid = (ParamGridBuilder().addGrid(regressor.isotonic, [True, False]).build())

        # Create the evaluators
        eval_rmse = RegressionEvaluator(metricName = "rmse")
        eval_r2 = RegressionEvaluator(metricName = "r2")
        
        # Prepara o Cross Validator
        crossval = CrossValidator(estimator = regressor, estimatorParamMaps = paramGrid, evaluator = eval_rmse, numFolds = 3)
        
        # Train the model using cross validation
        modelo = crossval.fit(dados_treino)
        
        # Extract the best model
        BestModel = modelo.bestModel

        # Summary of each model
        # Model metrics
        if tipo_algo in("DecisionTreeRegressor"):
            
            # Global Variable
            global DT_BestModel 
            DT_BestModel = modelo.bestModel
            
            # Predictions with test data
            previsoes_DT = DT_BestModel.transform(dados_teste)
            
            print('\033[1m' + "Decision Tree Model With Cross Validation:" + '\033[0m')
            print(" ")
            
            # Model evaluation
            resultado_teste_rmse = eval_rmse.evaluate(previsoes_DT)
            print('RMSE in Test:', resultado_teste_rmse)
        
            resultado_teste_r2 = eval_r2.evaluate(previsoes_DT)
            print('R2 Coefficient in Test:', resultado_teste_r2)
            print("")
        
        # Model metrics
        if tipo_algo in("RandomForestRegressor"):
            
            # global variable
            global RF_BestModel 
            RF_BestModel = modelo.bestModel
            
            # Predictions with test data
            previsoes_RF = RF_BestModel.transform(dados_teste)
            
            print('\033[1m' + "RandomForest Model With Cross Validation:" + '\033[0m')
            print(" ")
            
            # Model evaluation
            resultado_teste_rmse = eval_rmse.evaluate(previsoes_RF)
            print('RMSE in Test:', resultado_teste_rmse)
        
            resultado_teste_r2 = eval_r2.evaluate(previsoes_RF)
            print('R2 Coefficient in Test:', resultado_teste_r2)
            print("")
        
        # Model metrics
        if tipo_algo in("GBTRegressor"):

            # global variable
            global GBT_BestModel 
            GBT_BestModel = modelo.bestModel
            
            # Predictions with test data
            previsoes_GBT = GBT_BestModel.transform(dados_teste)
            
            print('\033[1m' + "Gradient-Boosted Tree (GBT) Model With Cross Validation:" + '\033[0m')
            print(" ")
            
            # Model evaluation
            resultado_teste_rmse = eval_rmse.evaluate(previsoes_GBT)
            print('RMSE in Test:', resultado_teste_rmse)
        
            resultado_teste_r2 = eval_r2.evaluate(previsoes_GBT)
            print('R2 Coefficient in Test:', resultado_teste_r2)
            print("")
            
        # Model metrics
        if tipo_algo in("IsotonicRegression"):

            # global variable
            global ISO_BestModel 
            ISO_BestModel = modelo.bestModel
            
            # Predictions with test data
            previsoes_ISO = ISO_BestModel.transform(dados_teste)
            
            print('\033[1m' + "Isotonic Model With Cross Validation:" + '\033[0m')
            print(" ")
            
            # Avaliação do modelo
            resultado_teste_rmse = eval_rmse.evaluate(previsoes_ISO)
            print('RMSE in Test:', resultado_teste_rmse)
        
            resultado_teste_r2 = eval_r2.evaluate(previsoes_ISO)
            print('R2 Coefficient in Test:', resultado_teste_r2)
            print("")
                    
        # List of columns to put in the summary dataframe
        columns = ['Regressor', 'Resultado_RMSE', 'Resultado_R2']
        
        # Make predictions with test data
        previsoes = modelo.transform(dados_teste)
        
        # Evaluate the model to save the result
        eval_rmse = RegressionEvaluator(metricName = "rmse")
        rmse = eval_rmse.evaluate(previsoes)
        rmse_str = [str(rmse)]
        
        eval_r2 = RegressionEvaluator(metricName = "r2")
        r2 = eval_r2.evaluate(previsoes)
        r2_str = [str(r2)]
         
        tipo_algo = [tipo_algo] 
        
        # Create the dataframe
        df_resultado = spark.createDataFrame(zip(tipo_algo, rmse_str, r2_str), schema = columns)
        
        # Write the result to the dataframe
        df_resultado = df_resultado.withColumn('Resultado_RMSE', df_resultado.Resultado_RMSE.substr(0, 5))
        df_resultado = df_resultado.withColumn('Resultado_R2', df_resultado.Resultado_R2.substr(0, 5))
        
        return df_resultado


"""
============================
RUN MACHINE LEARNING MODULE
============================
"""

# List of algorithms
regressores = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GBTRegressor(), IsotonicRegression()]

# List of columns and values
colunas = ['Regressor', 'Resultado_RMSE', 'Resultado_R2']
valores = [("N/A", "N/A", "N/A")]

# Prepare the summary table
df_resultados_treinamento = spark.createDataFrame(valores, colunas)

# training loop
for regressor in regressores:
    
    # For each regressor get the result
    resultado_modelo = func_modulo_ml(regressor)
    
    # Save the results
    df_resultados_treinamento = df_resultados_treinamento.union(resultado_modelo)

# Return lines other than N/A
df_resultados_treinamento = df_resultados_treinamento.where("Regressor!='N/A'")

# Print
df_resultados_treinamento.show(10, False)

# The GBT model showed the best overall performance and will be used in production.
# Making predictions with the trained model
# To make predictions with the trained model, let's prepare a record with new data.
# - Cement: 540
# - Blast Furnace Slag: 0
# - Fly Ash: 0
# - Water: 162
# - Superplasticizer: 2.5
# - Coarse Aggregate: 1040
# - Fine Aggregate: 676
# - Age: 28

# List of input values
values = [(540,0.0,0.0,162,2.5,1040,676,28)]

# Column names
column_names = dados.columns
column_names = column_names[0:8]

# Bind values to column names
novos_dados = spark.createDataFrame(values, column_names)

# We apply the same transformation applied in the data preparation to the age column.
novos_dados = novos_dados.withColumn("age", log("age") +1)

# Attribute list
lista_atributos = ["cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "age"]

# Create the vectorizer
assembler = VectorAssembler(inputCols = lista_atributos, outputCol = 'features')

# Convert data to vector
novos_dados = assembler.transform(novos_dados).select('features')

# Standardize the data (same transformation applied to training data)
novos_dados_scaled = scalerModel.transform(novos_dados)

# Select the resulting column
novos_dados_final = novos_dados_scaled.select('scaledFeatures')

# Rename the column (MLlib requirement)
novos_dados_final = novos_dados_final.withColumnRenamed('scaledFeatures','features')

# Predictions with new data using the best performing model
previsoes_novos_dados = GBT_BestModel.transform(novos_dados_final)

# Result
previsoes_novos_dados.show()
