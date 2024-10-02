import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Dataframe printing configuration
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

np.set_printoptions(precision=2, suppress=True)

# Arquivo de leitura classificacao MAXIMO
csv_path = "PETR3-DIARIO-MAXIMO.csv"

df = pd.read_csv(csv_path)

print("************** Dados **************")
print(df.tail())

# Remove time column from Dataframe
date_time = pd.to_datetime(df.pop('tempo'), format='%Y-%m-%d')

print("************** Dados sem data **************")
print(df.tail())

# Normalize os dados
# Padronizar o DataFrame
scaler = StandardScaler()
#df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_std = df

print("************** SHAPE DF **************")
print(df_std.shape)

print("************** Dados normalizados **************")
print(df_std.tail())

previsao = df_std.pop('previsao')

print("************** Dados sem a previsao **************")
print(df_std.tail())

print("************** Somente previsao **************")
print(previsao.tail())

# Divisao conjunto de treino e conjunto de testes
train_x, train_y = df_std.iloc[:-1], previsao.iloc[:-1]
test_x, test_y = df_std.iloc[-1], previsao.iloc[-1]

# Conjunto de treino
print("************** Conjunto de train_x **************")
print(train_x.tail())

print("************** Conjunto de train_y **************")
print(train_y.tail())

# Conjunto de testes
print("************** Conjunto de test_x **************")
print(test_x)

print("************** Conjunto de train_y **************")
print(test_y)

print("************** Numero de amostras **************")
print(train_x.shape[0])

print("************** Numero de carateristicas **************")
print(train_x.shape[1])

# Convertendo o DataFrame para um numpy array
array = df.to_numpy()
train_x_np = train_x.to_numpy()
train_y_np = train_y.to_numpy()
test_x_np  = test_x.to_numpy()

# Reshapando
train_x_np = train_x_np.reshape(train_x_np.shape[0], train_x_np.shape[1], 1)
test_x_np = test_x_np.reshape(1, test_x_np.shape[0], 1)

# Criando Modelo
model = Sequential()

# Primeira camada de convolução
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_x_np.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Segunda camada de convolução
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten antes das camadas densas
model.add(Flatten())

# Camadas densas
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Resumo do modelo
model.summary()


# Treinar o modelo
history = model.fit(train_x_np, train_y_np, epochs=1000, batch_size=32, verbose=1)

# Avaliar o modelo
#loss = model.evaluate(test_x, test_y)
#print(f"Loss (Erro Quadrático Médio): {loss}")

# Fazer previsões
predictions = model.predict(test_x_np)

# Inverter a normalização para obter os preços reais
print("************** PREVISOES **************")
print(predictions)

# Despadronizar o DataFrame
# test_despadronizado = scaler.inverse_transform(predictions)
test_despadronizado = predictions

#plt.figure(figsize=(14, 7))
plt.plot(test_despadronizado, color='blue', label='Preço Real')
plt.plot(predictions, color='red', label='Previsão de Preço')
plt.title('Previsão de Preços de Ações usando CNN')
plt.xlabel('Tempo')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()

print("PREVISÃO")
print(predictions)



