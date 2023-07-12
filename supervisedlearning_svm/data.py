import numpy as np
import os
import nltk
import pickle
import random
from collections import defaultdict
from bs4 import BeautifulSoup

#nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    """
    Remove stopwords, punctuation, and numbers from text.

    Args:
        text: article text

    Returns:
        Space-delimited and cleaned string
    """
    # tokenize text
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    tokens = [token.lower().strip() for token in tokens if token.lower() not in stopwords]

    # remove tokens without alphabetic characters (i.e. punctuation, numbers)
    tokens = [token for token in tokens if any(t.isalpha() for t in token)]

    return ' '.join(tokens)


def parse_article(tag):
    """
    Parse article and topics from Reuters corpus.

    Args:
        tag: Tag corresponding to article
    
    Returns:
        Tuple with article text and topics
    """
    topics = [str(d.text) for d in tag.find('topics').find_all('d')]
    article = tag.find('text')
    text = article.body.text if article.body else article.contents[-1]

    return (clean_text(text), topics)


def parse_from_sgml(datadir, split='ModApte'):
    """
    Parse Reuters news articles in SGML format.
    We are using the Modified Apte train/test split:
    * Training Set (9,603 docs): LEWISSPLIT="TRAIN";  TOPICS="YES"
    * Test Set (3,299 docs): LEWISSPLIT="TEST"; TOPICS="YES"

    Args:
        datadir: directory containing SGML files 
        split: train/test split to use (default: Modified Apte)

    Returns:
        List of train and test splits
    """
    train_set, test_set = [], []
    files = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f.endswith('.sgm')]
    for file in files:
        soup = BeautifulSoup(open(os.path.join(datadir, file), 'rb'), 'lxml')
        tags = soup.find_all('reuters', lewissplit='TRAIN', topics='YES')
        train_set += [parse_article(tag) for tag in tags]
        
        tags = soup.find_all('reuters', lewissplit='TEST', topics='YES')
        test_set += [parse_article(tag) for tag in tags]

    return train_set, test_set

def group_articles_by_topic(dataset):
    data = defaultdict(list)

    for text, topics in dataset:
        if len(topics) == 0:
            continue
        for topic in topics:
            data[topic].append(text)

    return data

def extract_tasks(train_set, test_set, topics, num_articles):
    # extract articles with given topic
    train_counts = [(topic, len(train_set[topic])) for topic in topics]
    test_counts = [(topic, len(test_set[topic])) for topic in topics]
    train_tasks = [(text, topic) for topic in topics for text in train_set[topic]]
    test_tasks = [(text, topic) for topic in topics for text in test_set[topic]]
    
    # calculate number of articles of each topic to return
    train_articles = min(train_counts, key=lambda x: x[1])[1]
    if train_articles > num_articles:
        train_articles = num_articles

    test_articles = min(test_counts, key=lambda x: x[1])[1]
    if test_articles > num_articles:
        test_articles = num_articles

    train_set, test_set = [], []
    for topic in topics:
        count = 0
        for text in train_tasks:
            if count >= train_articles:
                break
            if text[1] == topic:
                train_set.append(text)
                count += 1

        count = 0
        for text in test_tasks:
            if count >= test_articles:
                break
            if text[1] == topic:
                test_set.append(text)
                count += 1

    return train_set, test_set


def load_data(mode, datadir, topics, num_articles):
    """
    Load data.

    Args:
        mode: train or test
        datadir: directory containing SGML files 
    
    Returns:
        List of documents and list of topic labels
    """
    if not os.path.isfile(os.path.join(datadir, 'train_set.pkl')) or not os.path.isfile(os.path.join(datadir, 'test_set.pkl')):
        train_set, test_set = parse_from_sgml(datadir)
        with open(os.path.join(datadir, 'train_set.pkl'), 'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(datadir, 'test_set.pkl'), 'wb') as f:
            pickle.dump(test_set, f)
    else:
        train_set = pickle.load(open(os.path.join(datadir, 'train_set.pkl'), 'rb'))
        test_set = pickle.load(open(os.path.join(datadir, 'test_set.pkl'), 'rb'))

    train_set = group_articles_by_topic(train_set)
    test_set = group_articles_by_topic(test_set)
    train_set, test_set = extract_tasks(train_set, test_set, topics, num_articles)

    # deterministic shuffle with seed=0
    random.Random(0).shuffle(train_set)
    random.Random(0).shuffle(test_set)
    X, y = zip(*train_set) if mode == 'train' else zip(*test_set)
    return list(X), list(y)
