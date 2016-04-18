import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np

#global variable for list of genres
genre_list = ['animated', 'action', 'comedy', 'drama', 'family', 'fantasy', \
'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']

def load_frequencies(genre, num_words):
    '''
    For a given genre (text), create a frequency list of tuples of the num_words (int)
    most commonly used words in reviews of that genre. Return the list.
    '''
    filename = '../data/dict_' + genre + '.csv'
    freq_list = []
    with open(filename, 'r') as f:
        for i in xrange(num_words):
            line = f.readline().split(',')
            if line[1] == '':
                line[1] = '0'
            freq_list.append((line[0],int(line[1])))
    return freq_list

def paint_clouds(genre, cloud_words):
    '''
    For a given genre (text), paint a word cloud of at most cloud_words (int)
    words. Call the load_frequencies function to get a frequency list with 50
    more words in it than are needed for the word cloud, in case some don't fit.
    '''
    freq_list = load_frequencies(genre, cloud_words+50)
    wc = WordCloud(background_color = "white", max_words = cloud_words, \
    max_font_size = 40, random_state = 42)
    wc.fit_words(freq_list)
    fig = plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.title(genre)
    plt.show()
    filename = '../data/cloud_' + genre + '.png'
    fig.savefig(filename)

if __name__ == '__main__':
    for genre in genre_list:
        paint_clouds(genre,75)
