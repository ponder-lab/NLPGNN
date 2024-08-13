#! encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from nlpgnn.datas import Planetoid
from nlpgnn.metrics import Losess, Metric
from nlpgnn.models import GCNLayer
from nlpgnn.callbacks import EarlyStopping

from scripts.utils import write_csv
import timeit

tf.random.set_seed(10)

hidden_dim = 16
num_class = 6
drop_rate = 0.5
epoch = 200
early_stopping = 10
penalty = 5e-4

# cora, pubmed, citeseer
data = Planetoid(name="citeseer", loop=True, norm=True)

features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = data.load()


start_time = timeit.default_timer()
skipped_time = 0

model = GCNLayer(hidden_dim, num_class, drop_rate)

optimizer = tf.keras.optimizers.Adam(0.01)
crossentropy = Losess.MaskCategoricalCrossentropy()
accscore = Metric.MaskAccuracy()
stop_monitor = EarlyStopping(monitor="loss", patience=early_stopping)

total_loss = 0
loss_count = 0

total_accuracy = 0
accuracy_count = 0

epoch_count = 0

# ---------------------------------------------------------
# For train
for p in range(epoch):
    epoch_count += 1
    t = time.time()
    with tf.GradientTape() as tape:
        predict = model(features, adj, training=True)
        loss = crossentropy(y_train, predict, train_mask)
        loss += penalty * tf.nn.l2_loss(model.variables[0])

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    predict_v = model.predict(features, adj)
    loss_v = crossentropy(y_val, predict_v, val_mask)
    total_loss += loss_v
    loss_count += 1
    acc = accscore(y_val, predict_v, val_mask)
    total_accuracy += acc
    accuracy_count += 1
    print_time = timeit.default_timer()
    print("Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(p, loss_v.numpy(), acc, time.time() - t))
    skipped_time += timeit.default_timer() - print_time
    if stop_monitor(loss_v, model):
        break
# --------------------------------------------------------------------------------------

time = timeit.default_timer() - start_time - skipped_time
avg_loss = float(total_loss) / float(loss_count)
avg_accuracy = float(total_accuracy)/ float(accuracy_count)

# For test
predict_t = model.predict(features, adj)
acc = accscore(y_test, predict_t, test_mask)
loss = crossentropy(y_test, predict_t, test_mask)
print_time = timeit.default_timer()
print("Test Loss {:.4f} | ACC {:.4f}".format(loss.numpy(), acc))
skipped_time += timeit.default_timer() - print_time

write_csv(__file__, epoch_count, float(avg_accuracy), float(avg_loss), time)
