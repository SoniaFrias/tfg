import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Carga de los datos
train_data = np.load("./BLAST110/data/train.npz")
validation_data = np.load("./BLAST110/data/validation.npz")
test_data = np.load("./BLAST110/data/test.npz")

x_train, y_train = train_data['X'], train_data['y']
x_validation, y_validation = validation_data['X'], validation_data['y']
x_test, y_test = test_data['X'], test_data['y']

# Escalado
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validation = scaler.fit_transform(x_validation)
x_test = scaler.fit_transform(x_test)

# Clases
y_train = to_categorical(y_train, num_classes=2)
y_validation = to_categorical(y_validation, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(
    x_train,
    y_train, 
    validation_data=(x_validation, y_validation), 
    batch_size=256, 
    epochs=10
)

loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1) 
matriz = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['No-Blasto', 'Blasto'], digits=4)
print(report)
print("Matriz de Confusión:\n", matriz)