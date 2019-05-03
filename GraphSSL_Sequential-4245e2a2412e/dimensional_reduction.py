from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
import metric as metric_learn


def reduce_dimension(dimensional_reduction, embedding_size, X_train, Y_train, X):
    if(dimensional_reduction == 1):
        clf = LinearDiscriminantAnalysis()
        # clf = LocallyLinearEmbedding(n_jobs=8, n_components=17)
        clf.fit(X_train, Y_train)
        return (clf.transform(X_train), clf.transform(X))
    elif(dimensional_reduction == 2):
        lfda = metric_learn.LFDA(k=7, num_dims=embedding_size)
        lfda.fit(X_train, Y_train)
        return (lfda.transform(X_train), lfda.transform(X))


def transform_metric(metric_type, X_train, Y_train, X):
    learner = None
    if(metric_type == 1):
        learner = metric_learn.ITML_Supervised(
            verbose=True, max_iter=1000, gamma=1)
    elif(metric_type == 2):
        learner = LMNN(n_neighbors=20, max_iter=1000,  n_features_out=100)
    elif(metric_type == 3):
        learner = metric_learn.SDML_Supervised(verbose=True, use_cov=True)
    learner.fit(X_train, Y_train)
    return learner.transform(X)
