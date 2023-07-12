import argparse
import sys
import numpy as np
from data import load_data

def get_args():
    parser = argparse.ArgumentParser(description="Compute accuracy.")

    parser.add_argument("--datadir", type=str, required=True, help="The directory containing SGML files .")
    parser.add_argument("--topics", type=str, nargs=2, default=['earn', 'acq'], 
                        help="The two article topics to extract for binary text classification. \
                        The first argument is the topic you will try to predict.")
    parser.add_argument("--predictions-file", type=str, required=True, help="The predictions file to create. (Only used for testing.)")
    parser.add_argument("--num-articles", type=int, default=500, help="Number of articles per topic to extract.")

    args = parser.parse_args()

    return args

def compute_accuracy():
    args = get_args()
    _, y = load_data('test', args.datadir, args.topics, args.num_articles)
    true_labels = np.array([1 if c == args.topics[0] else 0 for c in y])

    with open(args.predictions_file, 'r') as f:
        predictions = f.readlines()
    predicted_labels = np.array([int(line.strip()) for line in predictions])

    if len(predicted_labels) != len(true_labels):
        print('Number of lines in two files do not match.')
        sys.exit()

    correct_mask = true_labels == predicted_labels
    num_correct = float(correct_mask.sum())
    total = correct_mask.size
    accuracy = num_correct / total

    return accuracy, num_correct, total

def main():
    accuracy, num_correct, total = compute_accuracy()
    print('Accuracy: %f (%d/%d)' % (accuracy, num_correct, total))

if __name__ == '__main__':
    main()
