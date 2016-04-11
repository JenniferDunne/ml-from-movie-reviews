from pymongo import MongoClient
import pandas as pd
import numpy as np
import indicoio


def init_mongo_client():
    '''
    Start the Pymongo client, access the movie_reviews database, access the movies
    collection, and return pointer. (later, we will access the reviews collection)
    '''
    client = MongoClient()
    db = client.movie_reviews
    coll = db.movies
    revcoll = db.reviews
    #revcoll = db.subset      # subset of <200 movie reviews for testing
    return coll, revcoll


def get_vectors():
    '''
    Iterate through the collection of movie reviews, sending them in batches of
    100 to Indico to create 300-feature vectors for each document. Add a field
    for the doc_vec to the movie review entries, as well as a field with the
    word_count of the review.
    NOTE: Because the Mongo cursor will time out before the entire database can
    be iterated through, this is set to grab 1000 items at a time from the database.
    '''
    movie_reviews = reviews_coll.find({'word_count':{'$exists':False}}, {'_id': 1, 'review': 1}, limit=1000)
    i = 0
    while movie_reviews.count() > 0:
        review_list = []
        id_list = []
        for movie_dict in movie_reviews:
            movie_review = movie_dict['review']
            movie_id = movie_dict['_id']
            review_list.append(movie_review)
            id_list.append(movie_id)
            # batch reviews in groups of 100 to send to Indico to create a document vector
            if len(id_list)>99:
                print "Sending reviews {0} to {1} to Indico for vectorizing".format(i, i+len(id_list)-1)
                results = indicoio.text_features(review_list)
                for j in xrange(len(id_list)):
                    this_id = id_list[j]
                    doc_vec = results[j]
                    reviews_coll.update_one({'_id':this_id}, {'$push': {'doc_vec': doc_vec}})
                    # while we're at it, determine the length of the review
                    reviews_coll.update_one({'_id':this_id}, {'$push': {'word_count': len(review_list[j].split())}})
                # after the vectors have been posted to the database, update the counter and batch lists
                i += len(id_list)
                id_list = []
                review_list = []
        # when the reviews have been iterated through, if it wasn't an even 100, batch the remaining reviews
        if len(id_list)>0:
            print "Sending reviews {0} to {1} to Indico for vectorizing".format(i, i+len(id_list)-1)
            results = indicoio.text_features(review_list)
            for j in xrange(len(id_list)):
                this_id = id_list[j]
                doc_vec = results[j]
                reviews_coll.update_one({'_id':this_id}, {'$push': {'doc_vec': doc_vec}})
                # while we're at it, determine the length of the review
                reviews_coll.update_one({'_id':this_id}, {'$push': {'word_count': len(review_list[j].split())}})
        movie_reviews = reviews_coll.find({'word_count':{'$exists':False}}, {'_id': 1, 'review': 1}, limit=1000)


if __name__ == '__main__':
    #Hide API before pushing to GitHub!!!
    indicoio.config.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    movies_coll, reviews_coll = init_mongo_client()
    get_vectors()
