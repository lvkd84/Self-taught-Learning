from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode, PCA
from sklearn.svm import SVC #, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import itertools
import numpy as np
import matplotlib.pyplot as plt

def reduce_dimensionality(examples, num_components):
    pca = PCA(n_components=num_components)
    return pca.fit_transform(examples)

def learn_basis_from_unlabeled_data(unlabeled_examples, num_components, alpha, max_iter):
    dic = MiniBatchDictionaryLearning(n_components=num_components, alpha=alpha, n_iter=max_iter)
    return dic.fit(unlabeled_examples).components_

def learn_representation_for_labeled_data(labeled_examples, dictionary, max_iter):
    return sparse_encode(labeled_examples, dictionary, max_iter=max_iter)

def SVM_classifier(examples, labels, epsilon=0.001, max_iter=-1, fold_num=3, EECS440=False):
    lamb_das = [0.01, 0.1, 0.5, 1, 3, 5]
    gammas = [0.001, 0.01, 0.1, 0.5, 1, 2, 3]
    degrees = [2,3]
    kernels = ['poly']#['rbf','poly','linear']
    param_grid = {'kernel':kernels,'C':lamb_das,'gamma':gammas, 'degree':degrees}
    estimator = SVC(tol=epsilon, max_iter=max_iter)
    if EECS440:
        fold_splitter = StratifiedKFold(fold_num, random_state=12345)
        #folds = fold_splitter.split(examples, labels)
        grid_search = GridSearchCV(estimator, param_grid, cv=fold_splitter, n_jobs=-1, verbose=10)
        grid_search.fit(examples, labels.ravel())
        best_estimator = grid_search.best_estimator_
        print('Accuracy', grid_search.best_score_)
        fold_accuracy = list()
        for train_index, test_index in fold_splitter.split(examples, labels):
            class_accuracy = np.zeros((10,))
            train_examples = examples[train_index]; train_labels = labels[train_index]
            test_examples = examples[test_index]; test_labels = labels[test_index]
            best_estimator.fit(train_examples, train_labels)
            predicted_labels = best_estimator.predict(test_examples)
            for idx in range(len(test_labels)):
                if test_labels[idx] == predicted_labels[idx]:
                    class_accuracy[test_labels[idx]] = class_accuracy[test_labels[idx]] + 1
            class_accuracy = class_accuracy / (len(examples) / (10*fold_num))
            fold_accuracy.append(class_accuracy)
        print(fold_accuracy)
        return fold_accuracy
    else:
        grid_search = GridSearchCV(estimator, param_grid, cv=fold_num, n_jobs=-1, verbose=10)
        grid_search.fit(examples, labels.ravel())
        return grid_search.best_estimator_

def run_evaluation(test_examples, test_labels, classifier, plot=False, title=None):
    predicted_labels = classifier.predict(test_examples)
    print(predicted_labels)
    accuracy = classifier.score(test_examples, test_labels.ravel())
    labels = np.array([0,1,2,3,4,5,6,7,8,9])
    cm = confusion_matrix(test_labels, predicted_labels, labels=labels)
    print("ACCURACY: ", accuracy)
    plot_confusion_matrix(cm, labels, title=title)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
