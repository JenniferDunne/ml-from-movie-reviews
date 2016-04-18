import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pickle


def build_dataframes():
    '''
    Import CSV files of movies and movie reviews into dataframes
    '''
    df_all_500 = pd.read_csv('../data/df500.csv')
    Xtr = np.loadtxt('../data/Xtr500.csv', delimiter=',')
    ytr = np.loadtxt('../data/ytr500.csv', delimiter=',')
    Xt = np.loadtxt('../data/Xt500.csv', delimiter=',')
    yt = np.loadtxt('../data/yt500.csv', delimiter=',')
    return df_all_500, Xtr, ytr, Xt, yt


def test_prediction(filename, features, target):
    '''
    Given the filename of a pickled estimator, unpickle the estimator, run the
    features through it, compare the results with the target, and return the
    results.
    '''
    estimator = pickle.load( open( filename, "rb" ) )
    res = estimator.predict_proba(features)[:,0]
    print "\nResults for {0}: \n".format(filename)
    print "\nRounded by .25\n"
    yrd = round_by(res, .25)
    print metrics.classification_report(target, yrd)
    print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))
    print "\nRounded by .1\n"
    yrd = round_by(res, .1)
    print metrics.classification_report(target, yrd)
    print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))
    return res


def test_grid(features, target):
    '''
    Given a list of models for each genre, run the features through the models to
    predict target labels, and compare the predictions to the true target labels.
    '''
    genre_list = ['animated', 'action', 'comedy', 'drama', 'family', 'fantasy', \
    'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
    ypred_mat = np.empty([target.shape[0], target.shape[1]])
    for i in xrange(target.shape[1]):
        filename = '../data/is_' + genre_list[i] + '.pkl'
        ypred = test_prediction(filename, features, target[:,i])
        for j, prob in enumerate(ypred):
            ypred_mat[j,i] = prob
    with open('../data/grid_pkl_500.txt','w') as f:
        f.write("Model rounded by .25\n")
        yrd = round_by(ypred_mat, .25)
        f.write( metrics.classification_report(target, yrd) )
        f.write( "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd)) )
        f.write("\nModel rounded by .3\n")
        yrd = round_by(ypred_mat, .3)
        f.write( metrics.classification_report(target, yrd) )
        f.write( "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd)) )
        f.write("\nModel rounded by .2\n")
        yrd = round_by(ypred_mat, .2)
        f.write( metrics.classification_report(target, yrd) )
        f.write( "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd)) )
        f.write("\nModel rounded by .1\n")
        yrd = round_by(ypred_mat, .1)
        f.write( metrics.classification_report(target, yrd) )
        f.write( "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd)) )


def build_multi_class(features, target):
    '''
    Initialize a multi-label multi-class classifier, then fit it.
    '''
    print "Building model..."
    model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 1000, random_state=42), n_jobs=-1)
    print "Training model...\n(This may take a while)"
    model.fit(features, target)
    return model


def test_multi_class(model, features, target):
    '''
    Given a multi-label multi-class classifier, test features, and a test target,
    score how well it does. Write the scores to a file.
    '''
    with open('../data/gboost_1000_trees.txt','w') as f:
        f.write("Creating predictions...\n")
        ypred = model.predict_proba(features)
        f.write("Model rounded by .25\n")
        yrd = round_by(ypred, .25)
        f.write( metrics.classification_report(target, yrd) )
        print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))
        f.write("\nModel rounded by .3\n")
        yrd = round_by(ypred, .3)
        f.write( metrics.classification_report(target, yrd) )
        print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))
        f.write("\nModel rounded by .2\n")
        yrd = round_by(ypred, .2)
        f.write( metrics.classification_report(target, yrd) )
        print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))
        f.write("\nModel rounded by .4\n")
        yrd = round_by(ypred, .4)
        f.write( metrics.classification_report(target, yrd) )
        print "Percent of misclassification: {}\n".format(metrics.zero_one_loss(target, yrd))


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


def make_matrix(df):
    '''
    Given a dataframe containing 'doc_vec_arr', and 'target' columns,
    return matrices of features and targets
    '''
    print "Dividing the data..."
    target = df['abr_targ']
    train = df['doc_vec_arr']
    train = pd.Series(train.values)
    features = np.matrix([r for r in train])
    y = pd.Series(target.values)
    ym = np.matrix([r for r in y])
    return features, ym


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



if __name__ == '__main__':
    df_all_500, Xtr500, ytr500, Xt500, yt500 = build_dataframes()
    model500 = build_multi_class(Xtr500, ytr500)
    test_multi_class(model500, Xt500, yt500)
    #test_grid(Xt500, yt500)
