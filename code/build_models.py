import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
    Helper function to abbreviate target lists. Removing 'crime', 'music', 'history',
    and 'sport' genres, because they're really more plot elements than actual genres.
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


def iterate_models(features, target):
    '''
    For every column in the target matrix, build a model to predict that target.
    (Parameters for models came from previous grid searches.)
    Pickle all the models for use in classify_genres program.
    '''
    genre_list = ['animated', 'action', 'comedy', 'drama', 'family', 'fantasy', \
    'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
    est_list = [700,700,700,700,700,500,700,700,500,500,700,700,700,700]
    depth_list = [5,4,5,6,4,5,3,4,7,5,5,5,4,3]
    for i in xrange(target.shape[1]):
        print "Building model for {} reviews".format(genre_list[i])
        model = GradientBoostingClassifier(n_estimators = est_list[i], \
        max_depth=depth_list[i], random_state=42)
        print "Training model"
        model.fit(features, np.array(target[:,i].reshape(1,-1))[0])
        print "Pickling model"
        filename = '../data/is_' + genre_list[i] + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)


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

if __name__ == '__main__':
    df_movies, df_reviews = build_dataframes()
    df_reviews = transform_word_count(df_reviews)
    df_reviews = transform_doc_vec(df_reviews)
    # create a joined database including all the movie info in the review record
    result = pd.merge(df_reviews, df_movies, how='left', on='title_url')
    result = transform_target(result)
    df_all_500 = create_sub_frame(result, 500)
    Xtr500, Xt500, ytr500, yt500 = test_train(df_all_500)
    iterate_models(Xtr500,ytr500)
    
