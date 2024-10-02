import pandas as pd

# Leitura do arquivo minima csv
df_min = pd.read_csv('ArquivosGerados\\resultado_minima.csv', header=None)

# Leitura do arquivo maxima csv
df_max = pd.read_csv('ArquivosGerados\\resultado_maxima.csv', header=None)

print(df_min.head())
print(df_max.shape)

med_min =  round(df_min.median(), 2)
med_max = round(df_max.median(), 2)

print(med_min[0])
print(med_max[0])

# Salvar resultados em um arquivos
arquivo1 = open("ArquivosGerados\\mediana_minima.txt", "a")
arquivo2 = open("ArquivosGerados\\mediana_maxima.txt", "a")

arquivo1.write(str(med_min[0]))
arquivo2.write(str(med_max[0]))

# Fechando arquivos
arquivo1.close()
arquivo2.close()