import sys

from self_taught_functions import *
from get_data import *

NUM_FOLD = 5
def main():
    if len(sys.argv) != 5:
        raise Exception("Inappropriate number of arguments. Require 4.")
    unlabeled_data_name = str(sys.argv[1])
    labeled_data_name= str(sys.argv[2])
    run_on_full_sample = int(sys.argv[3])
    EECS440 = int(sys.argv[4])

    if unlabeled_data_name == 'fashion_mnist':
        unlabeled_examples = get_fashion_mnist(200, unlabeled=True)
    elif unlabeled_data_name == 'mnist':
        unlabeled_examples = get_mnist(200, unlabeled=True)
    elif unlabeled_data_name == 'house_number':
        unlabeled_examples = get_house_number(200, unlabeled=True)

    if labeled_data_name == 'fashion_mnist':
        labeled_examples, labels = get_fashion_mnist(600)
        test_examples, test_labels = get_fashion_mnist(500, test=True)
    elif labeled_data_name == 'mnist':
        labeled_examples, labels = get_mnist(600)
        test_examples, test_labels = get_mnist(500, test=True)
    elif labeled_data_name == 'house_number':
        labeled_examples, labels = get_house_number(600)
        test_examples, test_labels = get_house_number(500, test=True)

    # Learn basis
    print("START BASIS")
    #unlabeled_examples = reduce_dimensionality(unlabeled_examples, 500)
    basis = learn_basis_from_unlabeled_data(unlabeled_examples, 200, 1, 1000)
    # Learn new representation
    print("START REPRESENTATION")
    #labeled_examples = reduce_dimensionality(labeled_examples, 500)
    new_labeled_examples = learn_representation_for_labeled_data(labeled_examples, basis, 1000)
    #test_examples = reduce_dimensionality(test_examples, 500)
    new_test_examples = learn_representation_for_labeled_data(test_examples, basis, 1000)

    if run_on_full_sample:
        if EECS440:
            # Learn a SVC with model selection
            print("START CLASSIFIER")
            classifier = SVM_classifier(new_labeled_examples, labels, epsilon=0.000001, fold_num=3, EECS440=True)
        else:
            # Learn a SVC with model selection
            print("START CLASSIFIER")
            classifier = SVM_classifier(new_labeled_examples, labels, epsilon=0.000001, fold_num=5)
            # Validation
            print("START EVALUATION")
            run_evaluation(new_test_examples, test_labels, classifier, plot=True, title=unlabeled_data_name + ' ' + labeled_data_name)
    else:
        pass

if __name__ == '__main__':
    main()
