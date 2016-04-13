import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pickle

def build_dataframes():
    '''
    Import CSV files of movies and movie reviews into dataframes
    '''
    df_movies = pd.read_csv('../data/movies.csv')
    df_reviews = pd.read_csv('../data/reviews.csv')
    return df_movies, df_reviews

def get_val_from_inside(term):
    '''
    Helper function to strip out the surrounding brackets and convert to integer
    '''
    return int(term[1:-1])

def get_arr_from_inside(term):
    '''
    Helper function to strip out the surrounding brackets and convert to an array of float
    '''
    term_list = term[2:-2].split(',')
    return np.array([float(t) for t in term_list])

def make_bool_list(l):
    '''
    Helper function to convert lists of 'true' and 'false' strings to integer 1 and 0.
    '''
    term_list = l[1:-1].split(',')
    return np.array([make_bool(t) for t in term_list])

def abbreviate_targ(l):
    '''
    Helper function to abbreviate target lists.
    '''
    term_list = [l[0], l[1], l[2], l[4], l[5], l[6], l[8], l[10], l[11], l[12], l[13], l[15], l[16], l[17]]
    return np.array(term_list)

def make_bool(term):
    '''
    Helper function to convert 'true' and 'false' strings to integer 1 and 0.
    '''
    if term=='true':
        return 1
    else:
        return 0

def transform_word_count(df):
    '''
    Given a DataFrame (df) with the column 'word_count', return a dataframe
    with the new column 'wd_ct' that has an integer value of that value.
    '''
    df['wd_ct'] = df['word_count'].apply(get_val_from_inside)
    return df

def transform_doc_vec(df):
    '''
    Given a DataFrame (df) with the string column 'doc_vec', return a dataframe
    with the new column 'doc_vec_arrr' that has an array of float for that value.
    '''
    df['doc_vec_arr'] = df['doc_vec'].apply(get_arr_from_inside)
    df.drop('doc_vec', axis=1)
    return df

def transform_target(df):
    '''
    Given a DataFrame (df) with the string column 'genre_vec', return a dataframe
    with the new column 'target' that has an array of boolean integers for that value.
    Then create a column called 'abr_targ' that has the abbreviated target array.
    '''
    df['target'] = df['genre_vec'].apply(make_bool_list)
    df['abr_targ'] = df['target'].apply(abbreviate_targ)
    return df

def make_target_matrix(target):
    '''
    Given a pandas series of numpy arrays, convert it to a 2d numpy make_target_matrix
    '''
    num_rows = len(target)
    num_cols = len(target[0])
    mat = np.matrix()


def create_sub_frame(df, n=500):
    '''
    Create sub-data-frame of only those reviews in df that are at least n (int) words
    '''
    if 'wd_ct' not in df.columns:
        df = transform_word_count(df)
    return df[df['wd_ct'] > n]


def iterate_grid(features, target):
    '''
    For every column in the target matrix, build a grid search to determine the
    best parameters for predicting that target. Combine all of the models in a
    list of models. Return the list of models.
    '''
    genre_list = ['action', 'animation', 'comedy', 'drama', 'family', 'fantasy', \
    'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
    model_list = []
    for i in xrange(len(target[0])):
        model = build_grid(features, target[:,i], genre_list[i])
        model_list.append(model)
    return model_list

def build_grid(features, target, genre):
    '''
    Build an individual grid search across varying parameters for a single genre.
    Return the best model.
    '''
    grid = GridSearchCV(GradientBoostingClassifier, {n_estimators: [100, 300, 500, 700], \
    max_depth: [3,4,5,6,7]}, n_jobs=-1, random_state=42)
    filename = '../data/grid_' + genre + '.txt'
    with open(filename) as f:
        f.write("The best estimator for " + genre + " was:\n")
        f.write(grid.best_params_)
        f.write("\nAnd a score of: {}".format(grid.best_score_))
    return grid.best_estimator_


def test_grid(model_list, features, target):
    '''
    Given a list of models for each genre, run the features through the models to
    predict target labels, and compare the predictions to the true target labels.
    '''
    ypred_mat = np.empty(len(target), len(target[0]))
    for i in xrange(len(target[0])):
        model = model_list[i]
        ypred = model.predict_proba(features)
        for j, prob in enumerate(ypred):
            ypred_mat[j,i] = prob
    with open('../data/grid_abbr_500.txt','w') as f:
        f.write("Model rounded by .25\n")
        yrd = round_by(ypred_mat, .25)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .3\n")
        yrd = round_by(ypred_mat, .3)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .2\n")
        yrd = round_by(ypred_mat, .2)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .4\n")
        yrd = round_by(ypred_mat, .4)
        f.write( metrics.classification_report(target, yrd) )


def build_multi_class(features, target):
    '''
    Initialize a multi-label multi-class classifier, then fit it.
    '''
    print "Building model..."
    model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 500, random_state=42), n_jobs=-1)
    print "Training model...\n(This may take a while)"
    model.fit(features, target)
    return model

def test_multi_class(model, features, target):
    '''
    Given a multi-label multi-class classifier, test features, and a test target,
    score how well it does. Write the scores to a file.
    '''
    with open('../data/gboost_abbr_500.txt','w') as f:
        f.write("Creating predictions...\n")
        ypred = model.predict_proba(features)
        f.write("Model rounded by .25\n")
        yrd = round_by(ypred, .25)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .3\n")
        yrd = round_by(ypred, .3)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .2\n")
        yrd = round_by(ypred, .2)
        f.write( metrics.classification_report(target, yrd) )
        f.write("\nModel rounded by .4\n")
        yrd = round_by(ypred, .4)
        f.write( metrics.classification_report(target, yrd) )

def round_by(predictions, n):
    '''
    Given an array of probabilistic predictions, return an array with
    every value higher than or equal to n a 1, and every value lower a 0.
    '''
    res_list = []
    for i in xrange(len(predictions)):
        round_list = []
        for j in xrange(len(predictions[i])):
            if predictions[i][j] >= n:
                round_list.append(1)
            else:
                round_list.append(0)
        res_list.append(round_list)
    return np.array(res_list)

def test_train(df):
    '''
    Given a dataframe containing 'doc_vec_arr', and 'target' columns,
    return matrices of features and targets for training and testing
    '''
    print "Dividing the data..."
    target = df['abr_targ']
    train = df['doc_vec_arr']
    Xtr, Xt, ytr, yt = train_test_split(train, target, test_size=.3, random_state=42)
    Xtr = pd.Series(Xtr.values)
    Xtrm = np.matrix([r for r in Xtr])
    Xt = pd.Series(Xt.values)
    Xtm = np.matrix([r for r in Xt])
    ytr = pd.Series(ytr.values)
    ytrm = np.matrix([r for r in ytr])
    yt = pd.Series(yt.values)
    ytm = np.matrix([r for r in yt])
    return Xtrm, Xtm, ytrm, ytm

def primary(df_movies, df_reviews):
    '''
    Given two dataframes, df_movies and df_reviews, clean and combine them.
    Then create sub dataframes based on review length. Build classifiers for
    each sub dataframe, and compare results.
    '''
    df_reviews = transform_word_count(df_reviews)
    df_reviews = transform_doc_vec(df_reviews)
    # create a joined database including all the movie info in the review record
    result = pd.merge(df_reviews, df_movies, how='left', on='title_url')
    result = transform_target(result)
    print "Creating subframes..."
    #df_all_150 = create_sub_frame(result, 150)
    #df_all_200 = create_sub_frame(result, 200)
    df_all_500 = create_sub_frame(result, 500)
    #Xtr150, Xt150, ytr150, yt150 = test_train(df_all_150)
    #Xtr200, Xt200, ytr200, yt200 = test_train(df_all_200)
    Xtr500, Xt500, ytr500, yt500 = test_train(df_all_500)
    #Xtrall, Xtall, ytrall, ytall = test_train(result)
    #print "For reviews of 500 words or more..."
    #model500 = build_multi_class(Xtr500, ytr500)
    #test_multi_class(model500, Xt500, yt500)
    #print "For reviews of 200 words or more..."
    #model200 = build_multi_class(Xtr200, ytr200)
    #test_multi_class(model200, Xt200, yt200)
    #print "For reviews of 150 words or more..."
    #model150 = build_multi_class(Xtr150, ytr150)
    #test_multi_class(model150, Xt150, yt150)
    #modelall = build_multi_class(Xtrall, ytrall)
    #test_multi_class(modelall, Xtall, ytall)
    model_list = iterate_grid(Xtr500, ytr500)
    test_grid(model_list, Xt500, yt500)

if __name__ == '__main__':
    df_movies, df_reviews = build_dataframes()
    primary(df_movies, df_reviews)
