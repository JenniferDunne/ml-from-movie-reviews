from bs4 import BeautifulSoup
import requests
from pymongo import MongoClient


def init_mongo_client():
    '''
    Start the Pymongo client, access the movie_reviews database, access the movies
    collection, and return pointer. (later, we will access the reviews collection)
    '''
    client = MongoClient()
    db = client.movie_reviews
    coll = db.movies
    revcoll = db.reviews
    return coll, revcoll


def get_films(genre, start_at=0):
    '''
    Given a GENRE (string), construct a URL for the most popular IMDB movies
    of that genre. START_AT (int) movie 0 for page 1, movie 51 for page 2, etc.
    Request the website at that URL and return results.
    '''
    url = 'http://www.imdb.com/search/title?at=0&genres=' + genre + '&sort=num_votes'
    if start_at == 0:
        result = requests.get(url+'&title_type=feature')
    else:
        result = requests.get(url+'&start='+str(start_at)+'&title_type=feature')
    return result


def get_reviews(filmid, start_at=0):
    '''
    Given a FILMID (string), construct a URL for the most popular IMDB reviews
    of that film. START_AT (int) review 0 for page 1, review 10 for page 2, etc.
    Request the website at that URL and return results.
    '''
    url = 'http://www.imdb.com/' + filmid + 'reviews?start=' + str(start_at)
    result = requests.get(url)
    return result


def loop_thru_genres():
    '''
    For each genre in a list of pre-selected IMDB movie genres, request 300 films
    from that genre, and process the results to load the title information in the
    movies collection. (Most films have multiple genres, so this is needed to get
    the average of 150 unique films per genre desired.)
    '''
    genres = ['action', 'animation', 'comedy', 'crime', 'family',
    'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance',
    'sci_fi', 'sport', 'thriller', 'war', 'western']
    #genres = ['sci_fi']
    for genre in genres:
        print "Scraping " + genre + " page 1"
        result = get_films(genre, 0)
        if result.status_code != 200:
            print 'WARNING failed to get page 1 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)
        print "Scraping " + genre + " page 2"
        result = get_films(genre, 51)
        if result.status_code != 200:
            print 'WARNING failed to get page 2 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)
        print "Scraping " + genre + " page 3"
        result = get_films(genre, 101)
        if result.status_code != 200:
            print 'WARNING failed to get page 3 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)
        print "Scraping " + genre + " page 4"
        result = get_films(genre, 151)
        if result.status_code != 200:
            print 'WARNING failed to get page 4 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)
        print "Scraping " + genre + " page 5"
        result = get_films(genre, 201)
        if result.status_code != 200:
            print 'WARNING failed to get page 5 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)
        print "Scraping " + genre + " page 6"
        result = get_films(genre, 251)
        if result.status_code != 200:
            print 'WARNING failed to get page 6 of {0} with code {1}'.format(genre, result.status_code)
            break
        else:
            process_titles(result.content)


def process_titles(html):
    '''
    Given raw HTML (str) results from a get_films search, pull all of the
    information that we care about for the 50 different film titles (1 page).
    Remove parenthesis from the film's year, and remove unwanted genres
    from the alphabetically ordered collection of genres. Then, if the title
    is not already in our collection of movies, add it.
    '''
    print "Processing scraped film titles"
    bad_genres = ['adventure', 'biography', 'documentary', 'drama', 'film-noir']
    soup = BeautifulSoup(html, 'html.parser')
    titles = soup.find_all('td', class_= "title")
    for t in titles:
        title_href = t.find('a')
        title_url = title_href['href']
        title = title_href.text
        year = t.find('span', {"class": "year_type"}).text[1:5]
        outline = t.find('span', {"class": "outline"})
        if outline <> None:
            outline = outline.text
        else:
            outline = ''
        genre_str = t.find('span', {"class": "genre"}).text
        genre_list = genre_str.lower().split(' | ')
        genre_list = [g for g in genre_list if g not in bad_genres]
        genre_str = ' | '.join(sorted(genre_list))
        title_dict = {"title_url":title_url,"title":title,"year":year, "outline":outline,"genre":genre_str}
        if not movies_coll.find_one({'title_url':title_dict['title_url']}):
            movies_coll.insert_one(title_dict)
        else:
            if title_url<>"/title/tt0103644/":
                print "Duplicate title: " + str(title_dict['title_url']) + title_dict['title']
            else:
                print "Duplicate title: /title/tt0103644/Alien3"


def loop_thru_movies():
    '''
    For each movie in the IMDB movies collection, request 40 reviews for that
    movie, and process the results to load the review in the reviews collection.
    '''
    movie_urls = movies_coll.find({}, {'title_url': 1, '_id': 0})
    for movie_dict in movie_urls:
        movie_url = movie_dict['title_url']
        if not reviews_coll.find_one({'title_url':movie_url}):
            # only get reviews for a movie if we haven't any reviews for it yet
            print "Scraping " + movie_url + " page 1"
            result = get_reviews(movie_url, 0)
            if result.status_code != 200:
                print 'WARNING failed to get page 1 of {0} with code {1}'.format(movie_url, result.status_code)
                break
            else:
                process_reviews(movie_url, result.content)
            print "Scraping " + movie_url + " page 2"
            result = get_reviews(movie_url, 10)
            if result.status_code != 200:
                print 'WARNING failed to get page 2 of {0} with code {1}'.format(movie_url, result.status_code)
                break
            else:
                process_reviews(movie_url, result.content)
            print "Scraping " + movie_url + " page 3"
            result = get_reviews(movie_url, 20)
            if result.status_code != 200:
                print 'WARNING failed to get page 3 of {0} with code {1}'.format(movie_url, result.status_code)
                break
            else:
                process_reviews(movie_url, result.content)
            print "Scraping " + movie_url + " page 4"
            result = get_reviews(movie_url, 30)
            if result.status_code != 200:
                print 'WARNING failed to get page 4 of {0} with code {1}'.format(movie_url, result.status_code)
                break
            else:
                process_reviews(movie_url, result.content)


def get_reviews_for(movie_url):
    '''
    It may happen that the system times out or otherwise fails to record all of the
    reviews for a particular movie. Remove any reviews for that movie from the
    MongoDB database and rerun this function to get only the reviews for that film.
    '''
    if not reviews_coll.find_one({'title_url':movie_url}):
        # only get reviews for a movie if we haven't any reviews for it yet
        print "Scraping " + movie_url + " page 1"
        result = get_reviews(movie_url, 0)
        if result.status_code != 200:
            print 'WARNING failed to get page 1 of {0} with code {1}'.format(movie_url, result.status_code)
        else:
            process_reviews(movie_url, result.content)
        print "Scraping " + movie_url + " page 2"
        result = get_reviews(movie_url, 10)
        if result.status_code != 200:
            print 'WARNING failed to get page 2 of {0} with code {1}'.format(movie_url, result.status_code)
        else:
            process_reviews(movie_url, result.content)
        print "Scraping " + movie_url + " page 3"
        result = get_reviews(movie_url, 20)
        if result.status_code != 200:
            print 'WARNING failed to get page 3 of {0} with code {1}'.format(movie_url, result.status_code)
        else:
            process_reviews(movie_url, result.content)
        print "Scraping " + movie_url + " page 4"
        result = get_reviews(movie_url, 30)
        if result.status_code != 200:
            print 'WARNING failed to get page 4 of {0} with code {1}'.format(movie_url, result.status_code)
        else:
            process_reviews(movie_url, result.content)


def has_alt(tag):
    '''
    This helper function is used with the Mongo find function to find rating image
    '''
    return tag.has_attr('alt')


def process_reviews(filmid, html):
    '''
    Given raw HTML (str) results from a get_reviews search for a specific FILMID (str),
    pull all of the information that we care about for the 10 different film reviews.
    '''
    print "Processing scraped film reviews"
    soup = BeautifulSoup(html, 'html.parser')
    all_reviews = soup.find('div', id='tn15content')
    divisions = all_reviews.find_all('div')
    # Keep only the divisions that contain header information for reviews
    for i in range(len(divisions)-1,-1,-1):
        d = divisions[i]
        if d.text == '\n\nWas the above review useful to you?\r\n\n\n\n\n':
            divisions.pop(i)
    # Keep only the paragraphs that contain body of reviews
    paragraphs = all_reviews.find_all('p')
    for i in range(len(paragraphs)-1,-1,-1):
        p = paragraphs[i]
        if p.text == '*** This review may contain spoilers ***':
            paragraphs.pop(i)
        elif p.text == 'Add another review':
            paragraphs.pop(i)
    for i, d in enumerate(divisions):
        headline = d.find('h2').text
        author_link = d.find('a')
        author = author_link['href']
        image = d.find(has_alt)    # make sure this is the rating, not the author
        try:
            rating = image['alt'][:-3]
        except:
            rating = None
        p = paragraphs[i]            # get the matching review paragraph
        review = p.text.replace('\n',' ')
        review_dict = {"title_url":filmid,"headline":headline,"author":author,"rating":rating,"review":review}
        reviews_coll.insert_one(review_dict)


if __name__ == '__main__':
    movies_coll, reviews_coll = init_mongo_client()
    #loop_thru_genres()
    #loop_thru_movies()
    get_reviews_for('/title/tt0064940/')
