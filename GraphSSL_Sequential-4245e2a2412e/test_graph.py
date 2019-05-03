from __future__ import division
from __future__ import print_function
import argparse
from graph_model import Graph_Model


import time
import tensorflow as tf

from utils import *
from gcn.models import GCN, MLP


### Parameters to build ####

# Vector model location
vector_location = '/stage/vectors/wiki.el.bin'
# each word embedding size after dimensional reduction
embedding_size = 7
# Window size of words
word_window = 3
# Train, Test and Dev location of corpus
train_loc = '/stage/ud/UD_Greek-GDT/el_gdt-ud-train.conllu'
test_loc = '/stage/ud/UD_Greek-GDT/el_gdt-ud-test.conllu'
dev_loc = '/stage/ud/UD_Greek-GDT/el_gdt-ud-dev.conllu'
# Type of dimensionality reduction 1)Linear 2)LFDA
dimensional_reduction = 1
# Dimensional Reduction Flag
metric_type = 1
# Load Graph from file
load_graph = 1
# Neareset neigbours count for graph
neighbors = 10
# location of unlabeled data
unlabeled_loc = None
# 1) FastText 2) Vec File 3) ELMo
embedding_type = 1
# 1) Concatenate 2) Mean
ngram_type = 1
# BIO
BIO = False


model = Graph_Model(word_window,
                    load_graph, neighbors, embedding_type, ngram_type, BIO)
# Add data to the model
model.add_data(train_loc, test_loc, dev_loc, vector_location,
               unlabeled_loc)
x, y, tx, ty, allx, ally, test_size, index = model.reduce_dimension(
    embedding_size, dimensional_reduction, metric_type)

model.build_graph()
graph = model.train()


'''


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 40,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    x, y, tx, ty, allx, ally, graph, index, test_size)
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy],
                    feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
'''
