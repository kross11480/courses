# Overview
The repository contains jupyter notebook (in Python 3) for analysing book ratings for 10000 most popular (i.e. with most ratings) books in Goodreads. All used libraries that are installed with the Anaconda distribution.

# Aim
The major aim is to find out how classics authors book ratings compare with popular books.

# Methodology:

The following step describes the methodology (CRISP-DM) used:
1. Problem Understanding

* How are ratings of the popular books?
* How classics authors book ratings compare with popular books? 
* Do voracious readers rate differently than infrequent readers? 
* Are there outliers, that is books which are judged good by voracious reader/rater community and bad by occasional ones and vice versa?

2. Data Understanding

For the analysis, a 2017 dataset has been cloned from [releases](https://github.com/zygmuntz/goodbooks-10k/releases). The Dataset contains
* ratings.csv: six million ratings (book_id, user_id, rating) for 10,000 popular books.
* books.csv: Information (icluding authors, title, avg_ratings, ...) on ten thousand most popular (with most ratings) books.
* classics.csv: Information on around 200 classics books.

3. Prepare Data

The dataset of 10K books is clean and had no missing or duplicate data.

4. Model Data

No modelling of data done.

5. Results

The analysis finds out that although many classics are also popular (i.e. with lots of ratings) but their average rating is significantly lower than that of popular books. Furthermore, the frequent reader tend to give a lower rating than infrequent reader. Mostly, the frequent and infrequent reader agree on the ratings. However, there are many outliers, i.e. books having high average rating but lower ratings from frequent readers.
Checkout the [article](https://medium.com/@hritam79/do-not-judge-a-book-by-its-rating-9a8681a1757e).

6. Deploy

Try out the [jupyter notebook](book_analysis.ipynb) for the analysis of the dataset.

# Acknowledgements:
Thanks to author [Zygmuntz](https://github.com/zygmuntz) for providing the dataset scraped from Goodreads.