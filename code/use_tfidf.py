import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from collections import defaultdict
from collections import Counter
import pickle

#global variable for list of genres
genre_list = ['animated', 'action', 'comedy', 'drama', 'family', 'fantasy', \
'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']

# global variables for all non alphanumeric characters to be used in clean function
contractions = re.compile(r"'|-")
symbols = re.compile(r'(\W+)', re.U)
numeric = re.compile(r'(?<=\s)(\d+|\w\d+|\d+\w)(?=\s)', re.I)
seps = re.compile(r'\s+')

def clean(text):
    '''
    Given a review (text), use the above defined global variables to add spaces
    around words, lower case words, replace apostrophes with hyphens for contractions,
    and convert all numbers into a single designated number, 000. These choices
    for cleaning came from Matt Taddy's Doc2Vec parser.*/*
    '''
    # first strip non-ascii characters, then use the above defined substitutions (order matters)
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    text = u' ' +  text.lower() + u' '
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = numeric.sub('000', text)
    text = seps.sub(' ', text)
    return text

def review_word_count(review, feature_words):
    '''
    Given a cleaned review (review), count all of the words in the review and
    create a dictionary listing the number of occurrences of words belonging to
    the set of feature_words.
    '''
    words = review.split()
    meaningful_words = [w for w in words if w in feature_words]
    return Counter(meaningful_words)

def load_frequencies(genre, num_words):
    '''
    For a given genre (text), create a frequency list of tuples of the num_words (int)
    most commonly used words in reviews of that genre. Return the list.
    '''
    filename = '../data/dict_' + genre + '.csv'
    freq_list = []
    with open(filename, 'r') as f:
        for i in xrange(num_words):
            line = f.readline().split(',')
            if line[1] == '':
                line[1] = '0'
            freq_list.append((line[0],int(line[1])))
    return freq_list

def just_words(num_words):
    '''
    For each genre in the genre_list global variable, load the first num_words (int)
    words and return the list of word lists.
    Note that this is intended to take the first few hundred or thousand words
    from a file that has tens of thousands of entries. If num_words approaches
    or exceeds the number of lines in the dictionary, use another method.
    '''
    top_words = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in xrange(14):
        genre = genre_list[i]
        word_list = top_words[i]
        filename = '../data/dict_' + genre + '.csv'
        with open(filename, 'r') as f:
            while len(word_list) < num_words:
                word = f.readline().split(',')[0]
                word_list.append(word)
        top_words[i] = word_list
    return top_words

def make_features():
    '''
    Get the top 25 words for each genre. Combine them into a set of the top
    words for all genres. Return feature words.
    '''
    top_words = just_words(37)
    pop_words = set(top_words[0])
    pop_words = pop_words.union(top_words[1],top_words[2],top_words[3])
    pop_words = pop_words.union(top_words[4],top_words[5],top_words[6])
    pop_words = pop_words.union(top_words[7],top_words[8],top_words[9])
    pop_words = pop_words.union(top_words[10],top_words[11],top_words[12],top_words[13])
    with open('../data/feature_words.txt','w') as f:
        f.write(str(pop_words))
    return pop_words

def review_to_tf(df, feature_space):
    '''
    Given a dataframe(df) with a 'review' column, calculate the term frequency
    of the feature space for each review.
    '''
    # Iterate through the reviews in the dataframe, cleaning, counting, and
    # building a feature vector based on the feature_space(list).
    vector_list = []
    for n, row in df.iterrows():
        if n % 1000 == 0:
            print "Now formatting review number {}".format(n)
        word_dict = review_word_count(clean(row['review']),feature_space)
        feature_vector = []
        for word in feature_space:
            feature_vector.append(int(word_dict[word]))
        vector_list.append(feature_vector)
    df['doc_vec_arr'] = pd.Series(vector_list)
    return df

def test_train(train):
    '''
    Given the train dataframe with target sentiment values, divide it into
    training and testing dataframes for features, with training and testing targets.
    This is used instead of the supplied test data, because that had no targets.
    '''
    print "Dividing the data..."
    target = train['abr_targ']
    Xtr, Xt, ytr, yt = train_test_split(train, target, test_size=.3, random_state=42)
    Xtr.reset_index(inplace=True)
    Xt.reset_index(inplace=True)
    ytr = pd.Series(ytr.values)
    yt = pd.Series(yt.values)
    return Xtr, Xt, ytr, yt

def build_multi_class(features, target):
    '''
    Initialize a multi-label multi-class classifier, then fit it.
    '''
    print "Building model..."
    model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=900, random_state=42, max_depth=5), n_jobs=-1)
    print "Training model...\n(This may take a while)"
    model.fit(features, target)
    print "Pickling model"
    filename = '../data/tunedgradient.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return model


def test_multi_class(model, features, target):
    '''
    Given a multi-label multi-class classifier, test features, and a test target,
    score how well it does. Write the scores to a file.
    '''
    with open('../data/gboost_from_tf.txt','w') as f:
        f.write("Creating predictions...\n")
        ypred = model.predict_proba(features)
        f.write("Model rounded by .25\n")
        yrd = round_by(ypred, .25)
        f.write( metrics.classification_report(target, yrd) )
        f.write("Zero-one loss: {}\n".format(metrics.zero_one_loss(target, yrd)))
        f.write("\nModel rounded by .3\n")
        yrd = round_by(ypred, .3)
        f.write( metrics.classification_report(target, yrd) )
        f.write("Zero-one loss: {}\n".format(metrics.zero_one_loss(target, yrd)))
        f.write("\nModel rounded by .2\n")
        yrd = round_by(ypred, .2)
        f.write( metrics.classification_report(target, yrd) )
        f.write("Zero-one loss: {}\n".format(metrics.zero_one_loss(target, yrd)))
        f.write("\nModel rounded by .4\n")
        yrd = round_by(ypred, .4)
        f.write( metrics.classification_report(target, yrd) )
        f.write("Zero-one loss: {}\n".format(metrics.zero_one_loss(target, yrd)))


def round_by(predictions, n):
    '''
    Given an array of probabilistic predictions, return an array with
    every value higher than or equal to n a 1, and every value lower a 0.
    '''
    res_list = []
    for i in xrange(len(predictions)):
        round_list = []
        if len(predictions.shape) > 1:    # handle both 1-d and 2-d arrays
            for j in xrange(len(predictions[i])):
                if predictions[i][j] >= n:
                    round_list.append(1)
                else:
                    round_list.append(0)
            res_list.append(round_list)
        else:
            if predictions[i] >= n:
                res_list.append(1)
            else:
                res_list.append(0)
    return np.array(res_list)

def make_bool_list(l):
    '''
    Helper function to convert string representation of array to list of 1 and 0.
    '''
    term_list = l[1:-1].split(' ')
    return np.array([float(t) for t in term_list])

def make_matrix(df):
    '''
    Given a dataframe containing 'doc_vec_arr', and 'abr_targ' columns,
    return matrices of features and targets
    '''
    print "Dividing the data..."
    target = df['abr_targ'].apply(make_bool_list)
    train = df['doc_vec_arr']
    train = pd.Series(train.values)
    features = np.matrix([r for r in train])
    y = pd.Series(target.values)
    ym = np.matrix([r for r in y])
    return features, ym


if __name__ == '__main__':
    df500 = pd.read_csv('../data/df500.csv')
    Xtr500, Xt500, ytr500, yt500 = test_train(df500)
    pop_words = make_features()
    feature_space = list(pop_words)
    Xtr500_tf = review_to_tf(Xtr500, feature_space)
    Xt500_tf = review_to_tf(Xt500, feature_space)
    Xtrm, ytrm = make_matrix(Xtr500_tf)
    Xtm, ytm = make_matrix(Xt500_tf)
    model_tf = build_multi_class(Xtrm, ytrm)
    test_multi_class(model_tf, Xtm, ytm)
