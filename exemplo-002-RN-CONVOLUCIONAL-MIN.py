import pandas as pd
import tensorflow as tf

EPOCAS = 5000

# Formatar impressão
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format

# Leitura do arquivo csv
df = pd.read_csv('Dados\\PETR3-DIARIO-MINIMO.csv')

# Retirar coluna tempo do dataframe
date_time = pd.to_datetime(df.pop('tempo'), format='%d.%m.%Y')

# Tamanho da amostra
n = len(df)

# Divide dados em treino e teste
tamanho_dados_teste = 1

treino = df[0: n - tamanho_dados_teste]
teste = df[n - tamanho_dados_teste:]

# Valores que serão plotados no final
df_teste_previsao = teste

# Número de características
num_features = df.shape[1]

# Normalização
treino_mean = treino.mean()
treino_std = treino.std()

treino = (treino - treino_mean) / treino_std
teste = (teste - treino_mean) / treino_std

# Redes Convolucionais
CONV_WIDTH = 1

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH,), activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

conv_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

# Separa atributos da previsão
feature_cols_x = ["abertura", "maxima", "minima", "fechamento", "volume", "media7", "media21", "desvio14"]
feature_cols_y = ["previsao"]

treino_x = treino[feature_cols_x].to_numpy()
treino_y = treino[feature_cols_y].to_numpy()

teste_x = teste[feature_cols_x].to_numpy()

# Transforma em 3 dimensões no qual se calcula o eixo y
treino_x = treino_x.reshape(treino_x.shape[0], 1, treino_x.shape[1])
teste_x = teste_x.reshape(teste_x.shape[0], 1, teste_x.shape[1])

history = conv_model.fit(treino_x, treino_y, epochs=EPOCAS)

preditos = conv_model.predict(teste_x)

# Desfazer a Normalização
preditos = preditos * treino_std["previsao"] + treino_mean["previsao"]

df_teste_previsao.loc[:, "tempo"] = date_time[-tamanho_dados_teste:].values
df_teste_previsao.loc[:, "preditos"] = preditos[:, 0]

# Salvar resultados em um arquivos
arquivo = open("ArquivosGerados\\resultado_minima.csv", "a")

for index, row in df_teste_previsao.iterrows():
    arquivo.write("CONV-MIN;" + str(row["tempo"]) + ";" + str(row["previsao"]) + ";" + str(row["preditos"]) + "\n")

arquivo.close()

# Imprime na tela
print("Valores que serão plotados no final")
print(df_teste_previsao[["tempo", "previsao", "preditos"]].tail())

