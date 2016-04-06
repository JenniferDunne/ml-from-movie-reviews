import numpy as np
import pandas as pd
from pattern.vector import Document, Model, TFIDF

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


def create_doc_list(df):
    '''
    Given a dataframe containing an 'id' column and a 'review' column, create a
    list of documents in Pattern.Vector Document format. Because of how the data
    is formatted in the dataframe, the id contains an extra quote at the beginning
    and end of the id which need to be stripped away.
    '''
    print "Creating a list of {} documents".format(len(df))
    doc_list = []
    for index, row in df.iterrows():
        d = Document(row['review'], threshold=1, name=row['id'][1:-1])
        doc_list.append(d)
    return doc_list


def create_model(doc_list):
    '''
    Given a list of documents in Pattern.Vector Document format, create a
    Pattern.Vector Model.
    '''
    print "Creating a TFIDF model for {} documents".format(len(doc_list))
    return Model(documents=doc_list, weight=TFIDF)


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
    train_docs = create_doc_list(train)
    test_docs = create_doc_list(test)
    extra_docs = create_doc_list(extra)
    full_docs = train_docs + test_docs + extra_docs
    m = create_model(full_docs)
    save_model(m, '../data/small_model')

if __name__ == '__main__':
    load_and_save()
