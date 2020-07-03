# Recommendation Engine

A starter recommendation engine to make recommendations of articles to user, based on dataset which contains interactions of users and articles in the IBM Watson Platform.

# Information
The python notebook has been tested with python 3.7.6 on amazon ec2 cluster. The notebook implements
- data exploration
- rank-based recommendation: articles with most eyeballs
- collaborative filtering: recommend articles used by similar users
- use learning. i.e. svd matrix factorization for recommending articles
- TODO: content-based recommendation
