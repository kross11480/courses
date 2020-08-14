# Overview
The repository contains jupyter notebook (in Python 3) for analysing book ratings for 10000 most popular (i.e. with most ratings) books in Goodreads. All used libraries that are installed with the Anaconda distribution.

# Aim
The major aim is to find out how classics authors book ratings compare with popular books.

# File
For the analysis, a 2017 dataset has been cloned from [releases](https://github.com/zygmuntz/goodbooks-10k/releases). The Dataset contains
* ratings.csv: six million ratings (book_id, user_id, rating) for 10,000 popular books.
* books.csv: Information (icluding authors, title, avg_ratings, ...) on ten thousand most popular (with most ratings) books.
* classics.csv: Information on around 200 classics books.

Try out the [jupyter notebook](book_analysis.ipynb) for the analysis of the dataset.

# Results:
The analysis finds out that although many classics are also popular (i.e. with lots of ratings) but their average rating is significantly lower than that of popular books. Checkout the [article]().

# Acknowledgements:
Thanks to author [Zygmuntz](https://github.com/zygmuntz) for providing the dataset scraped from Goodreads.