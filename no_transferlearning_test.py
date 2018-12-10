import sys

from self_taught_functions import *
from get_data import *

NUM_FOLD = 5
def main():
    if len(sys.argv) != 3:
        raise Exeption("Inappropriate number of arguments. Require 2.")
    labeled_data_name= str(sys.argv[1])
    run_on_full_sample = int(sys.argv[2])

    if labeled_data_name == 'fashion_mnist':
        labeled_examples, labels = get_fashion_mnist(2000)
        test_examples, test_labels = get_fashion_mnist(500, test=True)
    elif labeled_data_name == 'mnist':
        labeled_examples, labels = get_mnist(2000)
        test_examples, test_labels = get_mnist(500, test=True)
    elif labeled_data_name == 'house_number':
        labeled_examples, labels = get_house_number(2000)
        test_examples, test_labels = get_house_number(500, test=True)

    if run_on_full_sample:
        # Learn a SVC with model selection
        print("START CLASSIFIER")
        classifier = SVM_classifier(labeled_examples, labels, epsilon=0.000001, fold_num=5)
        # Validation
        print("START EVALUATION")
        run_evaluation(test_examples, test_labels, classifier, plot=True, title=labeled_data_name)
    else:
        pass

if __name__ == '__main__':
    main()
