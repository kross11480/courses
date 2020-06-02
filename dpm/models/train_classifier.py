import sys
import re
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, precision_recall_fscore_support

from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """load data from database
    Keyword Args:
        database_filepath -- path of sql database
    Returns: 
        X -- message 
        Y -- category data 
        categories -- category names 
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("DisasterData", con=engine)
    categories = list(df.columns.values)[4:]
    X = df.message.values 
    Y = df[categories]
    
    return X, Y, categories

def tokenize(text):
    """extract tokens from text 
    Keyword Args:
        text -- message
    Returns: 
        tokens -- list of tokens in text after 
                removing stopwords, stemming, and lemmatizing
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stopwords.words("english")]
    
    return tokens

    
def build_model():
    """build pipeline model
    Returns: 
        model -- sklearn pipeline model 
    """
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    return model

def optimize_model(model, X_train, Y_train):
    """ Function used for finding optimal parameters of pipeline.
    Note: Other parameters left out to avoid long execution time
    """
    parameters = {
        'vect__max_df': [0.5, 0.75, 1]
    }    

    cv = GridSearchCV(model, param_grid=parameters, cv=3, verbose=2)
    #GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)
    cv.fit(X_train, Y_train)
    print (cv.best_estimator_.steps)
    
def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate pipeline model on test data and prints 
    precision, recall, and fscore for each category
    
    Keyword Args:
        model -- trained sklearn pipeline model
        X_test -- test messages
        Y_test -- test categories to be compared against model prediction
    Returns: 
        None
    """
    y_pred = model.predict(X_test)
    for idx, label in enumerate(category_names):
        print (label)
        print(precision_recall_fscore_support(Y_test[label], y_pred[:,idx], average='weighted') )


def save_model(model, model_filepath):
    """dumps model as pickle file
    Keyword Args:
        model -- trained sklearn pipeline model
        model_filepath -- filepath of serialized model
    """
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        #print('Optimizing model...')
        #optimize_model(model, X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
