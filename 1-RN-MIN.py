import numpy as np
import pandas as pd
import tensorflow as tf

CONV_WIDTH = 50
MAX_EPOCHS = 1000

csv_path = "PETR3-DIARIO-MINIMO.csv"

df = pd.read_csv(csv_path)

date_time = pd.to_datetime(df.pop('tempo'), format='%Y-%m-%d')

#Split the data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[:n-1]
val_df = df[n-(2 * CONV_WIDTH):n-1]
test_df = df[n-CONV_WIDTH-1:]

# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# Data windowing
class WindowGenerator:
  def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


# 2. Split
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels


WindowGenerator.split_window = split_window


# 4. Create `tf.data.Dataset`s
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds


WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test


def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, mode='min', restore_best_weights=True)

  model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
  return history


#Janelas de dados
conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1, label_columns=['previsao'])

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

history_multi = compile_and_fit(multi_step_dense, conv_window)

"""The main down-side of this approach is that the resulting model can only be executed on input windows of exactly this shape."""
### Convolution neural network
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

"""Train and evaluate it on the ` conv_window` and it should give performance similar to the `multi_step_dense` model."""
history_conv = compile_and_fit(conv_model, conv_window)

# Recurrent neural network
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

history_lstm = compile_and_fit(lstm_model, conv_window)


def desnormalizar(valor):
    previsao_desnormalizado = valor * train_std['previsao'] + train_mean['previsao']

    return previsao_desnormalizado


previsao_multi = desnormalizar(multi_step_dense.predict(conv_window.test))
previsao_conv = desnormalizar(conv_model.predict(conv_window.test))
previsao_lstm = desnormalizar(lstm_model.predict(conv_window.test))

# Salvar resultados em um arquivos
arquivo = open("ArquivosGerados\\resultado_minima.csv", "a")

arquivo.write(str(previsao_multi[0][0][0]) + "\n")
arquivo.write(str(previsao_conv[0][0][0]) + "\n")
arquivo.write(str(previsao_lstm[0][0]) + "\n")

arquivo.close()