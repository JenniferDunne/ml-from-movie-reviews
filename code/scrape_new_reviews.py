from bs4 import BeautifulSoup
import requests
from pymongo import MongoClient

# global variable so it only needs to be loaded once
good_genres = ['animated', 'action', 'comedy', 'drama', 'family', 'fantasy', \
'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']

def init_mongo_client():
    '''
    Start the Pymongo client, access the movie_reviews database, access the movies
    collection, and return pointer. (later, we will access the reviews collection)
    '''
    client = MongoClient()
    db = client.movie_reviews
    coll = db.movies
    revcoll = db.reviews
    newrevcoll = db.new_reviews
    return coll, revcoll, newrevcoll


def get_genre_vector(genre_list):
    '''
    Given a list of genres that a movie belongs to, create a vector defining
    that movie genre combination.
    [Animation, Action, Comedy, Drama, Family, Fantasy, Horror,
    Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western]
    '''
    vector_list = []
    vector_list.append('animation' in genre_list)
    vector_list.append('action' in genre_list)
    vector_list.append('comedy' in genre_list)
    vector_list.append('drama' in genre_list)
    vector_list.append('family' in genre_list)
    vector_list.append('fantasy' in genre_list)
    vector_list.append('horror' in genre_list)
    vector_list.append('musical' in genre_list)
    vector_list.append('mystery' in genre_list)
    vector_list.append('romance' in genre_list)
    vector_list.append('sci-fi' in genre_list)
    vector_list.append('thriller' in genre_list)
    vector_list.append('war' in genre_list)
    vector_list.append('western' in genre_list)
    return vector_list

def get_films(start_date='2016-03-01', end_date='2016-04-18', start_at=0):
    '''
    Given a start_date (string, yyyy-mm-dd) and end_date (string, yyyy-mm-dd),
    construct a URL for the most popular IMDB movies released within that time.
    of that genre. START_AT (int) movie 0 for page 1, movie 51 for page 2, etc.
    Request the website at that URL and return results.
    '''
    url = 'http://www.imdb.com/search/title?countries=us&languages=en&release_date=' \
    + start_date + ',' + end_date + '&sort=moviemeter,asc'
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


def loop_thru_films():
    '''
    Get 100 films released during the period of time from when reviews were
    originally pulled and today.
    '''
    print "Scraping new films page 1"
    result = get_films('2016-03-01', '2016-04-17', 0)
    if result.status_code != 200:
        print 'WARNING failed to get page 1 of {0} with code {1}'.format('new films', result.status_code)
    else:
        process_titles(result.content)
    print "Scraping new films page 2"
    result = get_films('2016-03-01', '2016-04-17', 51)
    if result.status_code != 200:
        print 'WARNING failed to get page 2 of {0} with code {1}'.format('new films', result.status_code)
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
        genre_list = [g for g in genre_list if g in good_genres]
        genre_str = ' | '.join(sorted(genre_list))
        title_dict = {"title_url":title_url,"title":title,"year":year, "outline":outline, \
        "genre":genre_str, "genres":genre_list, "genre_vec":get_genre_vector(genre_list) }
        if not movies_coll.find_one({'title_url':title_dict['title_url']}):
            new_reviews_coll.insert_one(title_dict)
        else:
            print "Already reviewed: " + str(title_dict['title_url']) + title_dict['title']


def loop_thru_movies():
    '''
    For each movie in the IMDB movies collection, request 10 reviews for that
    movie, and process the results to load the longest review in the collection.
    '''
    movie_urls = new_reviews_coll.find({}, {'title_url': 1, '_id': 0})
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
    longest_review = 0
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
        review_dict = {"$set": {"headline":headline,"author":author,"rating":rating,"review":review}}
        review_len = len(review.split())
        if review_len > longest_review:
            longest_review = review_len
            new_reviews_coll.update_one({"title_url":filmid}, review_dict)


if __name__ == '__main__':
    movies_coll, reviews_coll, new_reviews_coll = init_mongo_client()
    loop_thru_films()
    loop_thru_movies()
