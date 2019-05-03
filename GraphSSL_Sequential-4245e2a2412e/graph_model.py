import fastText
import string
from math import ceil
from collections import defaultdict

from sklearn.semi_supervised import LabelPropagation
from annoy import AnnoyIndex
import numpy as np
import logging
from dimensional_reduction import reduce_dimension, transform_metric
from math import floor
from util import readEmbeddings
from label_propagation import LGC, CAMLP, HMN, PARW, OMNI
from scipy import sparse
from scipy.sparse import lil_matrix
from BIOValidation import compute_f1
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num = 300


class Graph_Model():
    '''
    Defines the graph model used
    '''

    def __init__(self, word_window, load_graph, neighbors, embedding_type=1, ngram_type=1, BIO=False):
        self.word_window = int(word_window)
        self.load_graph = load_graph
        self.neighbors = neighbors
        self.embedding_type = int(embedding_type)
        self.ngram_type = int(ngram_type)
        self.BIO = BIO

    def _compute_metrics(self, true_positives, false_positives, false_negatives):
        precision = float(true_positives) / \
            float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / \
            float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def add_data(self, train_loc, test_loc, dev_loc, vector_location, unlabeled_loc):
        elmo = None
        if(self.embedding_type == 2):
            self.embeddings, self.word2Idx = readEmbeddings(vector_location)
        elif(self.embedding_type == 1):
            self.model = fastText.load_model(vector_location)
        else:
            with open(vector_location) as file:
                elmo = [line.strip() for line in file]
        counter = -1

        train_size = 0
        test_size = 0
        dev_size = 0
        self.word_vector = []
        self.tag_dict = dict()
        self.tag_dict_real = dict()

        for corpus_loc in [train_loc, dev_loc, test_loc]:
            tagged_corpus = open(corpus_loc, 'r')

            tag_id = 0
            for line in tagged_corpus:
                split_list = line.split()
                counter += 1
                if '#' in line:
                    continue
                elif len(split_list) < 2:
                    self.word_vector.append((True))

                    continue
                # Get word and tag
                word = split_list[1]
                tag = split_list[3]

                if(tag in string.punctuation):
                    tag = 'O'

                # Remove IOB Tag
                invalidChars = set(string.punctuation.replace("-", ""))
                printable = set(string.printable)
                # if any(char in invalidChars for char in tag):
                #   continue

                if('-' in tag):
                    tag = tag.split('-')[1]

                vec = None
                if(self.embedding_type == 1):
                    vec = self.model.get_word_vector(word)
                elif(self.embedding_type == 2):
                    vec = self._get_embeddings(word)
                else:
                    vec = np.array([float(num2)
                                    for num2 in elmo[counter].split()[1:]])
                    if(len(vec) != 1024):
                        print(vec)

                # if(word in string.punctuation):
                    # vec = np.full((1024,), 1)
                # if(word.isdigit()):
                    # vec = np.full((1024,), 2)

                if word != '' and word != '\n':
                    if tag not in self.tag_dict:
                        print(tag_id, tag)
                        self.tag_dict_real[tag_id] = tag
                        self.tag_dict[tag] = tag_id
                        tag_id += 1
                    if(corpus_loc == train_loc):
                        train_size += 1
                    elif(corpus_loc == test_loc):
                        test_size += 1
                    else:
                        dev_size += 1
                    self.word_vector.append(
                        (vec, word, self.tag_dict[tag]))
                else:
                    self.word_vector.append(np.zeros((num,), dtype=np.int))
        self.train_size = train_size
        self.test_size = test_size
        self.dev_size = dev_size
        print("Train Size: %d" % (train_size))
        print("Dev Size: %d" % (dev_size))
        print("Test Size: %d" % (test_size))
        print("Total Size: %d" % (len(self.word_vector)))
        if(unlabeled_loc != None):
            unlabeled_file = open(unlabeled_loc)
            for word in unlabeled_file:
                if(self.embedding_type == 1):
                    if word != '\n':
                        self.word_vector.append(
                            (self.model.get_word_vector(word), word, -1))
                elif(self.embedding_type == 2):
                    self.word_vector.append(
                        (self._get_embeddings(word), word,  -1))

    def prepare_embedding(self):
        '''
        Prepare the n gram embedding
        word_window: number of word window combinations
        '''
        word_embedding = []
        single_word = []
        tags = []
        rng = int(floor(self.word_window / 2))
        for index in range(0, len(self.word_vector)):
            if(isinstance(self.word_vector[index], bool)):
                continue
            temp_vector = self.word_vector[index][0]
            tags.append(self.word_vector[index][2])
            if(self.word_window != 0):
                for i in range(-1 * rng, rng + 1):
                    if index + i > 0 and index + i < len(self.word_vector) and i != 0:
                        if(isinstance(self.word_vector[index + i], bool)):
                            vector_i = self.model.get_word_vector('</s>')
                        else:
                            vector_i = self.word_vector[index + i][0]

                        if(self.ngram_type == 1):
                            temp_vector = np.array(np.concatenate(
                                (temp_vector, vector_i)))
                        else:
                            temp_vector = np.array(np.add(
                                temp_vector, vector_i))
                    elif i != 0:
                        if(self.ngram_type == 1):
                            temp_vector = np.array(np.concatenate(
                                (temp_vector, np.zeros((num, ), dtype=int))))
                        else:
                            tt = np.zeros((num, ), dtype=float)
                            temp_vector = np.array(np.add(temp_vector, tt))
            if(self.ngram_type == 1):
                word_embedding.append(temp_vector)
            else:
                word_embedding.append(np.divide(temp_vector, 3))

            single_word.append(self.word_vector[index][1].split('\n')[0])
        print(len(word_embedding))
        # self.word_vector=None
        return (word_embedding, tags, single_word)

    def save_obj(self, obj, filename):
        import pickle
        with open(filename, 'wb') as outfile:
            pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)

    def reduce_dimension(self, embedding_size, dimensional_reduction, metric_type):
        '''
        Training based on data
        '''
        X, Y, self.X_real = self.prepare_embedding()
        X_train = X[:self.train_size + self.dev_size]
        Y_train = Y[:self.train_size + self.dev_size]
        Y_test = Y[self.train_size + self.dev_size:]
        self.X_real = self.X_real[self.train_size:]
        self.Y_train = np.array(Y_train)
        metric_type = int(metric_type)

        if(metric_type != 2):
            print('Reducing dimension from here')

            X_train, self.X = reduce_dimension(
                dimensional_reduction, embedding_size, X_train, Y_train, X)
            x = self.X[:self.train_size]
            tx = self.X[self.train_size + self.dev_size:self.train_size +
                        self.dev_size + self.test_size]
            # self.save_obj(X_train, 'ind.afr.allx')
            allx = np.concatenate(
                (self.X[:self.train_size + self.dev_size], self.X[self.train_size + self.dev_size + self.test_size:]))

        else:
            print('Not Reducing dimension from here')
            self.X = np.array(X)
        n_values = np.max(Y) + 1
        Y_all = np.eye(n_values)[Y]
        ally = np.concatenate(
            (Y_all[:self.train_size + self.dev_size], Y_all[self.train_size + self.dev_size + self.test_size:]))
        y = Y_all[:self.train_size]
        ty = Y_all[self.train_size + self.dev_size:self.train_size +
                   self.dev_size + self.test_size]
        index = np.arange(ally.shape[0],
                          ally.shape[0] + self.test_size).tolist()

        print('Y_all', Y_all.shape)
        self.Y = Y_train
        self.Y_test = Y_test
        return (x, y, tx, ty, allx, ally, self.dev_size, index)

    def build_graph(self):
        if(self.load_graph == 2):
            with open('dimension.txt', 'r') as f:
                self.t = AnnoyIndex(int(f.readline()), metric='euclidean')
                self.t.load('graph.ann')
                print('Graph file loaded from file')
        else:
            print('building graph')
            print(self.X.shape)
            self.t = AnnoyIndex(self.X.shape[1], metric='euclidean')
            for i in range(0, self.X.shape[0]):
                self.t.add_item(i, self.X[i])
            self.t.build(50)
            print('graph built')
            print('graph saving')
            # self.t.save('graph.ann')
            # with open('dimension.txt', 'w') as f:
            #    f.write('%d' % self.X.shape[1])

    def train(self):
        graph_list = defaultdict(list)
        total_size = self.X.shape[0]
        Graph = lil_matrix((total_size, total_size))
        for i in range(0, total_size):
            near = self.t.get_nns_by_item(i, self.neighbors * 2)
            for index in near:
                graph_list[i].append(index)
                Graph[i, index] = 1
        models = [LGC(graph=Graph, max_iter=2000, alpha=0.001)]
        f = open('result.txt', 'w')
        self.save_obj(graph_list, 'ind.afr.graph')
        for clf in models:
            print('Results')
            f.write('Results')
            f.flush()
            x_train = np.arange(0, self.train_size + self.dev_size)
            x_test = np.arange(
                self.train_size + self.dev_size, self.train_size + self.dev_size + self.test_size)
            clf.fit(x_train, self.Y_train)
            y_predict = clf.predict(x_test)
            count = 0
            tot = 0
            true_positives = dict()
            false_positives = dict()
            false_negatives = dict()
            if(self.BIO):
                print(compute_f1([y_predict[:10000]],
                                 [self.Y_test[:10000]], self.tag_dict_real))
            for i in range(0,  self.test_size):
                real = self.Y_test[i]
                if(self.Y_test[i] != -1):
                    tot += 1
                if(y_predict[i] == real):
                    count += 1
            acc = count / float(tot)
            print(acc)
        return graph_list

    def _get_embeddings(self, token):
        if token not in self.word2Idx:
            token = 'UNKNOWN_TOKEN'
        return self.embeddings[self.word2Idx[token]]

    def getCasingVocab(self):
        entries = ['PADDING', 'other', 'numeric', 'mainly_numeric',
                   'allLower', 'allUpper', 'initialUpper', 'contains_digit']
        return {entries[idx]: idx for idx in range(len(entries))}
