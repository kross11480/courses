import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ read/merge message and catgory data into pandas dataframe
    Keyword args:
        messages_filepath -- filepath containing disaster messages
        categories_filepath -- filepath containing disaster categories
    Returns:
        df -- pandas dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id') 
    return df


def clean_data(df):
    """ clean pandas dataframe by one-hot encoding and removing old categories
    Keyword args:
        df -- pandas dataframe
    Returns:
        df -- cleaned pandas dataframe after one-hot encoding  
    """
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframee
    row = categories.iloc[0].str.split('-').str[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df= df.drop(columns=['categories'])
    df = df.merge(categories, left_index=True, right_index=True)
    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """ save pandas dataframe into sqlite database
    Keyword args:
        df -- pandas dataframe
        database_filename: filepath of sql database
    Returns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    try:
        df.to_sql('DisasterData', engine, index=False)
    except ValueError:
        pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()