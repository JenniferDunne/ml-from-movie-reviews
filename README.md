# Machine Learning from Movie Reviews

The idea of this project is that there are many sources of consumer reviews which can be mined for data.
This data can then be used to either:
	a) prompt specific activities in response
	b) create reviews based on others' opinions

I decided to work with movie reviews, because there are many databases already created for them, and
the information in the reviews can be used to predict a number of different things about the movie,
such as whether it was a "good" or "bad" movie, or what genre the movie was. While these predictions
are not of particular value in and of themselves, they are placeholders, indicating the types of
predictions that Natural Language Processing of reviews can be used for -- both boolean and categorical (sentiment analysis and customer segmentation).

## Sentiment Prediction

Phase 1 of this project involved predicting the sentiment of a movie review. The goal of this phase was
to determine the best way to process the vocabulary of the reviews, to inform the vocabulary processing
of the genre prediction phase.

Methods used to process reviews were:
* Bag of words with a Random Forest
* Word2Vec (converting each word to a feature vector)
* Doc2Vec (converting entire document to a feature vector)
* [Pattern Sentiment](http://www.clips.ua.ac.be/pages/pattern-en#sentiment)
* [Indico Sentiment](https://indico.io/docs#sentiment)
* [Indico Sentiment_HQ](https://indico.io/docs#sentiment_hq)

Pattern is a Python module with built-in sentiment analysis functions. Indico is a proprietary model with
two APIs for sentiment analysis.

Method | Accuracy | Precision | Sensitivity | Notes
--------|---------|-----------|--------------|------
Bag of Words (5000 features, 100 tree, no stemming) | .836 | .835 | .837 | Fast base line case
Bag of Words (5000 features, 100 tree, Porter stem) | .835 | .841 | .827 | Stemming made it worse
Bag of Words (5000 features, 500 tree, no stemming) | .850 | .849 | .847 | 5x the trees is better
Bag of Words (5000 features, 500 tree, Porter stem) | .853 | .847 | .858 | Stemming helped a bit
Bag of Words (6000 features, 500 tree, no stemming) | .843 | .838 | .846 | More features made it worse
Bag of Words (6000 features, 500 tree, Porter stem) | .851 | .848 | .853 | Stemming helped a bit
Word2Vec (using defaults) | .819 | .809 | .835 | Took 2 hours, for worse results
Indico Sentiment API (parsing by sentence) | .891 | .928 | .850 | Great results, very fast
Indico Sentiment API (weighted by sentence length) | .881 | .919 | .837 | Weighted sentences was worse
Indico Sentiment API (extra space after punctuation) | .892 | .927 | .853 | A bit better
Indico Sentiment API (no sentence parsing) | .901 | .928 | .871 | The best so far
Pattern built-in (using .01 cutoff) | .699 | .635 | .941 | Very few false negatives
Pattern built-in (using .1 cutoff) | .764 | .757 | .781 | Recommended cutoff
Pattern built-in (using .11 cutoff) | .762 | .769 | .751 | Slightly worse than .1
Pattern built-in (using .09 cutoff) | .762 | .741 | .809 | Slightly worse than .1
Doc2Vec distributed bag of words | .828 | .832 | .823 | Better than Word2Vec
Doc2Vec distributed model - concatenated | .702 | .702 | .707 | Worse than Word2Vec
Doc2Vec distributed model - mean | .820 | .826 | .813 | Slightly better than Word2Vec
Indico Sentiment_HQ API (no sentence parsing) | .932 | .935 | .929 | The best sentiment analysis

As a result of this phase, I determined that the amount of text in the limited number of movie
reviews did not provide enough text context for robust vectorization, compared to the amount of
training that went into developing the Indico API. Therefore, for subsequent phases, I will use
the Indico feature generation function to create the 300-feature vectors for my documents before
building classifiers.

## Genre Prediction

Phase 2 of the project involved analyzing a movie review to predict what genre film
the reviewer was talking about.
Predicting the genre of a film from the words used in a review has real-world applications
in segmenting a customer base or determining which marketing persona someone most closely
resembles.

### Challenge 1: Choosing which genres to use

The first challenge came in deciding which genres to use. My data came from the
Internet Movie Database (IMDB), and their complete genre list includes:
Action, Adventure, Animation, Biography, Comedy, Crime,
Documentary, Drama, Family, Fantasy, Film-Noir, History,
Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport,
Thriller, War, and Western

IMDB treats documentaries differently from all other movies, with an entirely different
storage structure, so those were easily eliminated. I dropped Adventure as a category,
because it seemed to overlap with Action (to the point where they are often referred to
as action/adventure movies). Film-noir is more of a filming style than the actual genre --
if it's filmed in black and white, rains a lot, and has "gritty realism", it's film-noir, but
that says very little about what genre conventions the film has. Most film-noirs belong to
the mystery or drama genres. Similarly, Music, Sport, Crime, and History speak to plot
elements within the film -- usually the backdrop against which the comedy or drama plays out
-- rather than specific genres.

That left me with a list of 14 genres: Action, Animation, Comedy, Drama, Family, Fantasy,
Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, and Western.

Using web scraping, BeautifulSoup, MongoDB, and PyMongo, I pulled a total of 3323 films,
getting the most popular 400 films from each genre. (Because most movies have more than
one genre, there was significant overlap in movie lists, which is how I ended up with 3323.)
I also originally pulled movies from the history, music, crime, and sport genres, and did
not pull movies from the drama genre. There still ended up being about twice as many dramas
as comedies.

### Challenge 2: Representing genres

My first thought is that I would make a single text string of the genres in a particular
movie, and categorize reviews in that fashion. After all, how many combinations of genres
could there really be? Romantic comedy, sci-fi action, mystery thriller... shouldn't be too
many. Turns out, there were over 500 different mixes. One film was categorized in eight
different genres! (The Great Race was listed as an Action, Adventure, Comedy, Family, Musical, Romance, Sport, Western film. I had to watch it just to see what it was, and it promptly became my
husband's favorite film.)

I first attempted to limit genres to a maximum of 4 per film, and combine films that
were unique combinations of genres. (For example, Mulan was the only animated family,
fantasy, musical, war movie. Dropping the "war" genre put it in the same category as many
other recent Disney musicals.) However, I ran into "outlier" movies, such as Perfect Blue,
which was an animated horror, mystery, thriller. Should they be combined with other films
that they really were not similar to? Removed as outliers that distort the data?

Even with these combinations, there were still hundreds of combinations. Even worse, since
so many films had unique combinations, odds are really good that future films would have
unique combinations as well, meaning my model would be doomed to fail when predicting
these new films. Clearly, this was not the best strategy.

Instead, I revised my strategy to predict a 14-element genre vector. The higher the first
value, the more likely the film was animated, etc.

### Challenge 3: Subsets of reviews or full dataset

In Phase 1, I determined that the 300-feature document vectors from Indico were the
best at predicting sentiment analysis. A feature vectorization API from them allowed
me to convert any document into its feature vector to use as an input to a predictive
model.

My first question was whether my predictions would be better using a smaller number
(19,914) of 500+ word-count reviews, a greater number (129,809) of any word-count reviews,
or a moderate number (72,691) of moderate length (200+ word-count) reviews. Would more
training data be better, or higher quality training data?

Number of Reviews |	Word Length	| F1 Score
------------------|-------------|----------
19,914 | 500+ | .57
72,691 | 200+ | .56
129,809 | All Reviews | .54

I used an F1 score to compare the results, because it evenly weighted both Accuracy
and Recall, providing a more balanced picture of the overall effectiveness of the
prediction. (As a level-set, if I were to randomly guess the genres for a particular
movie, given the distribution of film genres in my sample set, my accuracy would be
less than 20%.)	The difference between 200+ word reviews and 500+ word reviews was not
very much, however it clearly illustrated the trend that shorter reviews provided
less information. For subsequent tests, I would use the 500+ word reviews subset.

### Challenge 4: Type of predictive model

I conducted my initial comparisons using a Random Forest with default settings.
This had the advantage of being reasonably simple, fast, and not overly sensitive.
It also let me start from the same place as my Sentiment Analysis models, so I had
a good sense of the difference in abilities of the two different models.

I then switched to a Gradient Boost model, using the default settings from SKLearn,
with a OneVsRestClassifier wrapper.
That raised the results to an F1 score of .63.

I had hoped to compare the current darling of Kaggle competitions, XGBoost, however
I ran into too many technical challenges to be able to implement it within the short
time frame allowed for this project. (The auto-install for Windows has been disabled,
and trying three different work arounds that people swore made it work on their systems
all failed for me. Attempting to install it on my AWS instance destroyed the instance,
to the point where it was unrecoverable and I had to create a new one.)

I then tried to optimize predictors using GridSearchCV. I recorded the parameters that
yielded the best performing predictors for each genre, and created 14 separate predictors
that could be combined to yield an overall prediction. Even though the GridSearch results
indicated that prediction accuracy ranged between .75 and .95, I was unable to replicate
those results. Again, the short time-frame allowed for the project prevented me from
following up on this promising direction.

I settled for using the parameter analysis from the GridSearch to optimize my single
GradientBoostingClassifer. I adjusted the maximum depth of trees, the number of trees,
the learning rate, and even the loss function (I played around with using the
hamming distance as a loss function, or the zero-one loss). I was able to tune my
model to achieve an F1 score of .67 ... over 3 1/2 times as good as random chance.

## Word Clusters

I used TF-IDF analysis to determine which words were semi-unique to specific genres.
For example, "Disney" is a common word in reviews for both animated and family films, but not
in horror films.

The word clouds indicated which genres had significant overlap -- such as animated
and family -- as well as which were almost completely different -- horror films and
war movies shared the single word "bloody" among their top differentiating words.

![Animated Word Cloud](https://github.com/JenniferDunne/capstone/blob/master/data/cloud_animated.png)
![Family Word Cloud](https://github.com/JenniferDunne/capstone/blob/master/data/cloud_family.png)

![Horror Word Cloud](https://github.com/JenniferDunne/capstone/blob/master/data/cloud_horror.png)
![War Word Cloud](https://github.com/JenniferDunne/capstone/blob/master/data/cloud_war.png)

## Next Steps

I would like to build an app that allows someone to identify a new movie on IMDB
and run it through the models. I have already written code to perform web Scraping
to identify films released since the original web scraping, find the longest reviews
for those films, and prepare the reviews for being run against the model.
