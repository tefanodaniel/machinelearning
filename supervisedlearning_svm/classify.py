""" Main file. This is the starting point for your code execution.
You shouldn't have to make any code changes here.
"""
import os
import pickle
import argparse

import numpy as np

import models
import kernels
from data import load_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--datadir", type=str, required=True, help="The directory containing SGML files .")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--kernel", type=str, required=True, choices=["ngram", "tfidf"], help="Kernel function to use. Options include: ngram or tfidf.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create (for training) or load (for testing).")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")
    parser.add_argument("--topics", type=str, nargs=2, default=['earn', 'acq'], 
                        help="The two article topics to extract for binary text classification. \
                        The first argument is the topic you will try to predict.")
    parser.add_argument("--num-articles", type=int, default=500, help="Number of articles per topic to extract.")

    parser.add_argument("--pegasos-lambda", type=float, default=1e-3,
                    help="Model learning rate")
    parser.add_argument("--train-epochs", type=int, default=5,
                    help="Number of epochs to train the model")
    parser.add_argument("--ngram-length", type=int, default=3, help="Length of ngram sequences.")

    args = parser.parse_args()

    return args


def check_args(args):
    mandatory_args = {'datadir', 'mode', 'model_file', 'predictions_file', 'kernel', 'topics'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception('Arguments that we provided are now renamed or missing. If you hand this in, you will get 0 points.')

    if args.mode.lower() == "test":
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def test(args):
    """ Make predictionsz over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _ = load_data(args.mode.lower(), args.datadir, args.topics, args.num_articles)
    model = pickle.load(open(args.model_file, 'rb'))

    # compute kernel matrix
    if args.kernel.lower() == 'ngram':
        kernel = kernels.NgramKernel(ngram_length=args.ngram_length)
    elif args.kernel.lower() == 'tfidf':
        kernel = kernels.TFIDFKernel(X=model.support_vectors, X_prime=X)
    else:
        raise Exception("Kernel argument not recognized")
    kernel_matrix = kernel.compute_kernel_matrix(X=model.support_vectors, X_prime=X)


    # predict labels for dataset
    preds = model.predict(X=X, kernel_matrix=kernel_matrix)
    
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X, y = load_data(args.mode.lower(), args.datadir, args.topics, args.num_articles)
    y = np.array([1 if c == args.topics[0] else 0 for c in y])

    # build the appropriate model
    model = models.KernelPegasos(nexamples=len(X), lmbda=args.pegasos_lambda)

    # compute kernel matrix
    if args.kernel.lower() == 'ngram':
        kernel = kernels.NgramKernel(ngram_length=args.ngram_length)
    elif args.kernel.lower() == 'tfidf':
        kernel = kernels.TFIDFKernel(X=X)
    else:
        raise Exception("Kernel argument not recognized")
    kernel_matrix = kernel.compute_kernel_matrix(X=X)

    # Run the training loop
    for epoch in range(args.train_epochs):
        model.fit(X=X, y=y, kernel_matrix=kernel_matrix)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


if __name__ == "__main__":
    args = get_args()
    check_args(args)

    if args.mode.lower() == 'train':
        train(args)
    elif args.mode.lower() == 'test':
        test(args)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
