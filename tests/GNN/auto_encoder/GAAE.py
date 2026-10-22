#! encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from nlpgnn.datas import Planetoid
from nlpgnn.models import GAAELayer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nlpgnn.callbacks import EarlyStoppingScale

from scripts.utils import write_csv
import timeit

hidden_dim = 300
drop_rate = 0.5
epoch = 100

# cora, pubmed, citeseer
data = Planetoid(name="citeseer", loop=True, norm=True)

features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = data.load()

train_index = np.argwhere(train_mask == 1).reshape([-1]).tolist()
valid_index = np.argwhere(val_mask == 1).reshape([-1]).tolist()
test_index = np.argwhere(test_mask == 1).reshape([-1]).tolist()

start_time = timeit.default_timer()
skipped_time = 0

class GAAE(tf.keras.Model):
    def __init__(self, hidden_dim, lamb=1, **kwargs):
        super(GAAE, self).__init__(**kwargs)
        self.lamb = lamb
        self.hidden_dim = hidden_dim
        self.model = GAAELayer(hidden_dim, num_layers=2)

    @tf.function
    def call(self, node_embeddings, adjacency_lists, training=True):
        edge_sources = adjacency_lists[0][:, 0]  # [M]
        edge_targets = adjacency_lists[0][:, 1]  # [M]
        hidden_embeddings, reconstruct_embedding = self.model(node_embeddings, adjacency_lists, training)
        # The reconstruction loss of node features
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(node_embeddings - reconstruct_embedding, 2))))
        # The reconstruction loss of the graph structure
        s_emb = tf.nn.embedding_lookup(hidden_embeddings, edge_sources)
        r_emb = tf.nn.embedding_lookup(hidden_embeddings, edge_targets)
        structure_loss = -tf.math.log(tf.sigmoid(tf.reduce_sum(s_emb * r_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)
        loss = features_loss + self.lamb * structure_loss
        return loss, hidden_embeddings

    @tf.function
    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)


model = GAAE(hidden_dim)

optimizer = tf.keras.optimizers.Adam(0.1)

total_loss = 0
loss_count = 0

total_accuracy = 0
accuracy_count = 0

# ---------------------------------------------------------
# For train
hidden_embeddings=0
test_features = 0
test_y = 0
loss_v = 0
for p in range(epoch):
    t = time.time()
    with tf.GradientTape() as tape:
        loss, _ = model(features, adj, training=True)
        if p == 0: model.summary()
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    loss_v, hidden_embeddings = model.predict(features, adj)
    total_loss += loss_v
    loss_count += 1
    hidden_embeddings = hidden_embeddings.numpy()

    train_features = hidden_embeddings[train_index]
    train_y = np.argmax(y_train[train_index], -1)

    valid_features = hidden_embeddings[valid_index]
    valid_y = np.argmax(y_val[valid_index], -1)

    clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
    clf.fit(train_features, train_y)

    predict_y = clf.predict(valid_features)
    report_v = classification_report(valid_y, predict_y, digits=4, output_dict=True)
    acc = report_v["accuracy"]
    total_accuracy += acc
    accuracy_count += 1
    print_time = timeit.default_timer()
    print("EPOCH {:.0f} loss {:.4f} ACC {:.4f} Time {:.4f}".format(p,loss_v, acc, time.time()-t))
    skipped_time += timeit.default_timer() - print_time

train_features = hidden_embeddings[train_index]
train_y = np.argmax(y_train[train_index], -1)

test_features = hidden_embeddings[test_index]
test_y = np.argmax(y_test[test_index], -1)

clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
clf.fit(train_features, train_y)
predict_y = clf.predict(test_features)

report = classification_report(test_y, predict_y, digits=4, output_dict=True)
acc = report["accuracy"]
print("Test: Loss {:.5f} Acc {:.5f}".format(loss_v, acc))

time = timeit.default_timer() - start_time - skipped_time
avg_loss = float(total_loss) / float(loss_count)
avg_accuracy = float(total_accuracy)/ float(accuracy_count)

write_csv(__file__, epoch, float(avg_accuracy), float(avg_loss), time)
