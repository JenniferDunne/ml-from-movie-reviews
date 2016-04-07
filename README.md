# Machine Learning from Movie Reviews

The idea of this project is that there are many sources of consumer reviews which can be mined for data. 
This data can then be used to either:
	a) prompt specific activities in response
	b) create reviews based on others' opinions

I decided to work with movie reviews, because there are many databases already created for them, and 
the information in the reviews can be used to predict a number of different things about the movie, 
such as whether it was a "good" or "bad" movie, or what genre the movie was. While these predictions
are not of particular value in and of themselves, they are placeholders, indicating the types of
predictions that Natural Language Processing of reviews can be used for -- both boolean and categorical.

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
