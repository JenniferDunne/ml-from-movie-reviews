import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
from sklearn.cross_validation import train_test_split

# global variable for set of stop words, plus punctuation marks
stops = set(stopwords.words("english"))
stops = stops.union(set(['.',',','!','?',':',';','...','(',')','/','"','.)',').','")',')"']))

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

def review_to_dict(review):
    '''
    Given a cleaned review (review), count all of the words in the review and
    create a dictionary listing the number of occurrences of those words.
    '''
    words = review.split()
    meaningful_words = [w for w in words if not w in stops]
    return Counter(meaningful_words)

def find_words_by_genre(df):
    '''
    Given a dataframe (df) containing a 'review' column and an 'abr_targ' column,
    count the number of word occurrences for each genre across all reviews.
    Return a list of genre dictionaries.
    '''
    # First, build the list of genre_dicts, each of which is a defaultdict which
    # has a value of 0 for new words, and a key value pair of 'genre_dict' and
    # the name of the particular genre being counted in that dictionary.
    genre_dicts = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), \
    defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), \
    defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for i in xrange(14):
        genre_dicts[i]['genre_dict'] = genre_list[i]
    # Next, iterate through the reviews in the dataframe, cleaning, counting, and
    # adding their words to the appropriate dictionaries.
    for n, row in df.iterrows():
        if n % 1000 == 0:
            print "Now counting review number {}".format(n)
        word_dict = review_to_dict(clean(row['review']))
        targ_vals = row['abr_targ'][1:-1].split()
        for i, val in enumerate(targ_vals):
            if val == '1':
                for k,v in word_dict.items():
                    genre_dicts[i][k] += v
    # Now, sort the complete dictionaries by descending value and write to csv file
    # Be sure to remove the name of the dictionary before sorting, since it is not
    # a numeric value.
    for d in genre_dicts:
        filename = '../data/dict_' + d['genre_dict'] + '.csv'
        del d['genre_dict']
        with open(filename, 'w') as f:
            for w in sorted(d, key=d.get, reverse=True):
                f.write( w + ',' + str(d[w]) + '\n')

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
        filename = '../data/Xdict_' + genre + '.csv'
        with open(filename, 'r') as f:
            while len(word_list) < num_words:
                word = f.readline().split(',')[0]
                word_list.append(word)
        top_words[i] = word_list
    return top_words

def find_common_words():
    '''
    Get the top 5000 words for each genre. Find the words that are in the top
    5000 for all genres. Write those to a text file, so they can be used as
    domain-specific stop words.
    '''
    top_words = just_words(5000)
    pop_words = set(top_words[0])
    pop_words = pop_words.intersection(top_words[1],top_words[2],top_words[3])
    pop_words = pop_words.intersection(top_words[4],top_words[5],top_words[6])
    pop_words = pop_words.intersection(top_words[7],top_words[8],top_words[9])
    pop_words = pop_words.intersection(top_words[10],top_words[11],top_words[12],top_words[13])
    with open('../data/common_words.txt','w') as f:
        f.write(str(pop_words))

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

if __name__ == '__main__':
    # include the domain-specific stop words created by the find_common_words function
    with open('../data/common_words.txt','r') as f:
        common_words = f.read()
        common_words = common_words[5:-2].split(', ')
        common_words = [w[1:-1] for w in common_words]
        common_word_set = set(common_words)
    stops = stops.union(common_word_set)
    df500 = pd.read_csv('../data/df500.csv')
    Xtr500, Xt500, ytr500, yt500 = test_train(df500)
    find_words_by_genre(Xtr500)
    
