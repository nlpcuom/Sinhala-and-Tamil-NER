import pickle
import gzip
import numpy as np


def write_to_file(obj, file_name):
    pickle.dump(obj, open(file_name, 'wb'))


def read_from_file(file_name):
    return pickle.load(open(file_name, 'rb'))


def readEmbeddings(embeddingsPath):
    """
    Reads the embeddingsPath.
    :param embeddingsPath: File path to pretrained embeddings
    :return:
    """

    neededVocab = {}
    # :: Read in word embeddings ::
    print("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")

    embeddingsDimension = None

    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print(
                "ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    # Extend embeddings file with new tokens
    embeddings = np.array(embeddings)
    print('Embeddings read')
    return embeddings, word2Idx
