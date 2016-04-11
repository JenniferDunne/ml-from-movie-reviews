from pymongo import MongoClient
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests


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

def get_genre_vector(genre_list):
    '''
    Given a list of genres that a movie belongs to, create a vector defining
    that movie genre combination.
    [Animation, Action, Comedy, Crime, Drama, Family, Fantasy, History, Horror,
    Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western]
    '''
    vector_list = []
    vector_list.append('animation' in genre_list)
    vector_list.append('action' in genre_list)
    vector_list.append('comedy' in genre_list)
    vector_list.append('crime' in genre_list)
    vector_list.append('drama' in genre_list)
    vector_list.append('family' in genre_list)
    vector_list.append('fantasy' in genre_list)
    vector_list.append('history' in genre_list)
    vector_list.append('horror' in genre_list)
    vector_list.append('music' in genre_list)
    vector_list.append('musical' in genre_list)
    vector_list.append('mystery' in genre_list)
    vector_list.append('romance' in genre_list)
    vector_list.append('sci-fi' in genre_list)
    vector_list.append('sport' in genre_list)
    vector_list.append('thriller' in genre_list)
    vector_list.append('war' in genre_list)
    vector_list.append('western' in genre_list)
    return vector_list


def get_film(title_url):
    url = 'http://www.imdb.com' + title_url
    result = requests.get(url)
    soup = BeautifulSoup(result.content, 'html.parser')
    genre_all = soup.find('div', {"itemprop": "genre"}).text
    genre_list = genre_all[10:].split('|')
    genre_list = [g.strip().lower() for g in genre_list]
    genre_list = sorted(genre_list)
    genre_str = ' | '.join(genre_list)
    genre_vector = get_genre_vector(genre_list)
    movies_coll.update_one({'title_url':title_url}, \
    {'$set': {'genre_vec': genre_vector, 'genres': genre_list, 'genre': genre_str}})


def loop_thru_movies():
    movie_urls = movies_coll.find({'genre_vec':{'$exists':False}}, {'title_url': 1, '_id': 0})
    i = 0
    for movie_dict in movie_urls:
        movie_url = movie_dict['title_url']
        get_film(movie_url)
        if(i % 100 == 0):
            print "Status: Now on movie %d" % (i)
        i += 1


if __name__ == '__main__':
    movies_coll, reviews_coll = init_mongo_client()
    loop_thru_movies()
