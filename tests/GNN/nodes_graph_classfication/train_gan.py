#! encoding:utf-8
"""
@Author:Kaiyin Zhou
"""
import time
import tensorflow as tf
from nlpgnn.datas import Planetoid
from nlpgnn.metrics import Losess, Metric
from nlpgnn.models import GATLayer
from nlpgnn.callbacks import EarlyStopping

from scripts.utils import write_csv
import timeit

tf.random.set_seed(10)  # 随机选择的
hidden_dim = 8  # 8*heads=64
num_class = 7
drop_rate = 0.6
epoch = 1000
patience = 100
penalty = 0.0005  # for cora and citeseer
# penalty = 0.001  # for pubmed

# cora, pubmed, citeseer
loader = Planetoid(name="cora", loop=True, norm=True)

features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = loader.load()

start_time = timeit.default_timer()
skipped_time = 0

model = GATLayer(hidden_dim=hidden_dim, num_class=num_class, dropout_rate=drop_rate)

optimizer = tf.keras.optimizers.Adam(0.005)
crossentropy = Losess.MaskCategoricalCrossentropy()
accscore = Metric.MaskAccuracy()

total_loss = 0
loss_count = 0

total_accuracy = 0
accuracy_count = 0

# ---------------------------------------------------------
# For train
for p in range(epoch):
    t = time.time()
    with tf.GradientTape() as tape:
        predict = model(features, adj, training=True)
        loss = crossentropy(y_train, predict, train_mask)
        loss += tf.add_n([tf.nn.l2_loss(v) for v in model.variables
                          if "bias" not in v.name]) * penalty
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    predict_v = model.predict(features, adj)
    loss_v = crossentropy(y_val, predict_v, val_mask)
    total_loss += loss_v
    loss_count += 1
    acc_v = accscore(y_val, predict_v, val_mask)
    total_accuracy += acc_v
    accuracy_count += 1
    print_time = timeit.default_timer()
    print("Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(p, loss_v.numpy(), acc_v, time.time() - t))
    skipped_time += timeit.default_timer() - print_time
# --------------------------------------------------------------------------------------
# For test
predict_t = model.predict(features, adj)
acc = accscore(y_test, predict_t, test_mask)
loss = crossentropy(y_test, predict_t, test_mask)
print_time = timeit.default_timer()
print("Test Loss {:.4f} | ACC {:.4f}".format(loss.numpy(), acc))
skipped_time += timeit.default_timer() - print_time

time = timeit.default_timer() - start_time - skipped_time
avg_loss = float(total_loss) / float(loss_count)
avg_accuracy = float(total_accuracy)/ float(accuracy_count)

write_csv(__file__, epoch, float(avg_accuracy), float(avg_loss), time)
