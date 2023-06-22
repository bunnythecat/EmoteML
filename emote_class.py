import pickle
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from lib.transformer import Encoder, CosineDecay
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np

# gloval settings
save_path = "chat_20220718"
data_path = "chat_data"
maxlen = 40
data_size = 1813
vocab_size = 3086
# declare hyper-parameters
batch_size = 128
num_layers = 5
num_heads = 10
dff = 100
min_lr = 1E-7
dff2 = 28
dropout_rate = 0.1

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

emote_file = "emotes.txt"

with open(emote_file, 'r') as f:
    emotes = f.read().split(',')

with open(data_path + "/embedding_matrix.p", "rb") as f:
    embedding_matrix = pickle.load(f)


def get_column(matrix, i):
    return [row[i] for row in matrix]


y_preds = 0
# Cross validation with 4 folds, load each split training data and perform the cross validation
for i in range(4):
    model_save_path = save_path + "/" + str(i) + "/"
    with open("{}/train_{}.p".format(data_path, str(i)), "rb") as f:
        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(f)

    initial_bias = []
    class_weights = []
    for i in range(len(emotes)):
        neg, pos = np.bincount(get_column(y_train, i))
        initial_bias.append(np.log(pos / neg))
        class_weights.append((1 / pos) * ((pos + neg) / 2.0))

    class_weights = dict(enumerate(class_weights))
    print(initial_bias)
    print(class_weights)

    output_bias = tf.keras.initializers.Constant(initial_bias)
    inputs = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    encoder = Encoder(num_layers=num_layers, d_model=100, num_heads=num_heads, dff=dff, input_vocab_size=vocab_size,
                      maximum_position_encoding=maxlen, em_weights=embedding_matrix, rate=dropout_rate)

    x = encoder(inputs, training=True, mask=None)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(dff2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(len(emotes), activation='softmax', bias_initializer=output_bias,
                                    name="prediction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Class")
    model.summary()
    warmup_steps = data_size // batch_size * 10
    learning_rate = CosineDecay(
        min_lr=min_lr, max_lr=min_lr * 6000, warmup_steps=warmup_steps
    )
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-9)
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='acc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(curve='PR', name='prc'),
    ]
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer,
                  metrics=metrics)

    ckpt = model_save_path + "ckpt"

    model_callbacks = [
        callbacks.ModelCheckpoint(
            ckpt,
            monitor='val_loss', save_best_only=True, verbose=True, save_weights_only=True),
        callbacks.EarlyStopping(monitor='val_tp', patience=20, mode='max', restore_best_weights=True)
    ]
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs=200,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        verbose=1,
        callbacks=model_callbacks)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    hist.to_csv(model_save_path + "log.csv")

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.plot(hist['epoch'], hist['loss'], label='Train loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig(model_save_path + "history.png", dpi=300)

    # test
    results = model.evaluate(x_test, y_test, verbose=1)
    results = dict(zip(model.metrics_names, results))
    print(results)
    rslt = pd.DataFrame(data=results, index=[0])
    rslt.to_csv(model_save_path + "result.csv")

    y_pred = model.predict(x_test)
    # add predicted probability distribution to the soft voting
    y_preds += y_pred
    with open(model_save_path + "test_predictions.p", "wb") as f:
        pickle.dump([y_test, y_pred], f)

# soft voting
y_preds = y_preds / 4
print(y_preds[0])
m = tf.keras.metrics.AUC()
m.update_state(y_test, y_preds)
m1 = tf.keras.metrics.Precision()
m1.update_state(y_test, y_preds)
m2 = tf.keras.metrics.Accuracy()
m2.update_state(y_test, y_preds)
print("AUC-ROC: ", m.result().numpy())
print("Precision: ", m1.result().numpy())
print("Accuracy: ", m2.result().numpy())

with open("test.csv", 'r') as f:
    test_lines = f.read().split('\n')

X_test = []
y_test = []
y_pred = []
y_true = []
for line in test_lines:
    X_test.append(line.split(',')[0])
    y_test.append(line.split(',')[1])

for y in y_preds:
    y_pred.append(emotes[np.argmax(y)])

for y in y_test:
    y_true.append(emotes[np.argmax(y)])

with open(save_path + "/test_results.csv", 'w') as f:
    for x, y, z in zip(X_test, y_test, y_pred):
        f.write(",".join([x, y, z]) + "\n")
