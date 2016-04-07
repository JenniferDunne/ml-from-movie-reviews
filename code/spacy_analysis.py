import numpy as np
import pandas as pd
from spacy.en import English
from bs4 import BeautifulSoup
from spacy.tokens.doc import Doc


def load_data():
    '''
    Load the IMDB datasets into three categories -- train, test, and extra
    Because the datasets were set up for a competition, only the train datasets
    has target labels. The other datasets are additional words to be used in
    creating vocabularies only.
    '''
    print "Loading the data..."
    train = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3 )
    extra = pd.read_csv('../data/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    return train, test, extra

def remove_html(df):
    '''
    Given a dataframe containing a 'review' column containing raw reviews, clean
    the reviews by adding a space after punctuation, collapsing ellipses, and
    running through BeautifulSoup to get rid of all html tags. Return a list of
    the cleaned reviews.
    '''
    cleaned_reviews=[]
    print "Cleaning {} reviews".format(len(df['review']))
    for review in df['review']:
        # get rid of any html in the review
        review_text = BeautifulSoup(review, "lxml").get_text()
        # remove the opening and closing quotes around the review.
        review_text = review_text[1:-1]
        # add a space after every period, question mark, and exclamation point
        # then collapse ellipses
        review_text = review_text.replace('.','. ')
        review_text = review_text.replace('?','? ')
        review_text = review_text.replace('!','! ')
        review_text = review_text.replace('. . . ','...')
        cleaned_reviews.append(review_text)
    return cleaned_reviews


def create_doc_list(df):
    '''
    Given a dataframe containing an 'id' column and a 'review' column, create a
    list of tuples containing the id of the review, and the text of the review
    in spacy Doc format, which automatically divides the texts into sentences
    and individual tokens, and the id of the review. Because of how the data is
    formatted in the dataframe, the id contains an extra quote at the beginning
    and end of the id which need to be stripped away.
    '''
    print "Creating a list of {} documents".format(len(df))
    doc_list = []
    for index, row in df.iterrows():
        doc_list.append((row['id'][1:-1], nlp(row['review'])))
    return doc_list




def save_model(model, filename):
    '''
    Given a Pattern.Vector model, save it to filename. Include path.
    '''
    print "Saving model to file {}".format(filename)
    model.save(filename, update=False)


def load_model(filename):
    '''
    Given a path/filename, load the Pattern.Vector model from that filename.
    '''
    print "Loading model from file {}".format(filename)
    return Model.load(filename)

def load_and_save():
    '''
    Loads the IMDB datasets, creates a model from them, then saves the model
    '''
    train, test, extra = load_data()
    train['review'] = remove_html(train)
    test['review'] = remove_html(test)
    extra['review'] = remove_html(extra)
    train_docs = create_doc_list(train)
    test_docs = create_doc_list(test)
    extra_docs = create_doc_list(extra)
    full_docs = train_docs + test_docs + extra_docs
    print "Doc lists complete"
    doc_vec = doc_list[0][1].vector()
    doc_vec2 = doc_list[-1][1].vector()
    print "Length of document vector for first document is:\n{}\n".format(len(doc_vec))
    print "Length of document vector for last document is:\n{}\n".format(len(doc_vec2))
    print "Similarity between these two vectors is: {}".format(doc_list[0][1].similarity(doc_list[-1][1]))


if __name__ == '__main__':
    nlp = English()
    load_and_save()
