import sys

from multitask_feature_learning_functions import *
from get_data import *

NUM_FOLD = 5

def main():
    if len(sys.argv) < 3:
        raise Exeption("Inappropriate number of arguments. Need at least 2.")
    run_on_full_sample = int(sys.argv[1])

    # Get data
    num_tasks = int(sys.argv[2])
    task_examples = list(); task_labels = list()
    task_test_examples = list(); task_test_labels = list()
    for i in range(num_tasks):
        labeled_data_name = str(sys.argv[i+3])
        if labeled_data_name == 'fashion_mnist':
            labeled_examples, labels = get_fashion_mnist(2000)
            test_examples, test_labels = get_fashion_mnist(500, test=True)
        elif labeled_data_name == 'mnist':
            labeled_examples, labels = get_mnist(2000)
            test_examples, test_labels = get_mnist(500, test=True)
        elif labeled_data_name == 'house_number':
            labeled_examples, labels = get_house_number(2000)
            test_examples, test_labels = get_house_number(500, test=True)
        task_examples.append(labeled_examples); task_labels.append(labels)
        task_test_examples.append(test_examples); task_test_labels.append(test_labels)

    # Learn the representation
    W = learn_representations(task_examples, task_labels, gamma=2.0, epsilon=0.000001)

    # Run evaluation
    if run_on_full_sample:
        for i in range(num_tasks):
            run_evaluation(W, i, task_test_examples[i], task_test_labels[i])
    else:
        pass

if __name__ == '__main__':
    main()
