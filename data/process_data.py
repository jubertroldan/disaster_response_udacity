import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    parameters: 
        messages_filepath --> messages file location
        categories_filepath --> categories file location
    output:
        merged messages, categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """ Data cleaning process 
    parameters:
    df --> input dataframe to clean
    output:
    clean dataframe
    """
    
    # Split `categories` into separate category columns.  
    categories = df['categories'].str.split(';', 36,expand=True)
    row = categories.loc[1]
    category_colnames = list(row.str.split('-',expand=True)[0])
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
        
    # Replace `categories` column in `df` with new category columns.
    
    df.drop(columns=['categories'],inplace=True)
    
    df = df.merge(categories,left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)
    
    # removing related = 2 in the cleaning stage as this is just 0.007% of the data
    df = df[~(df['related']=='2')]
    return df

    

def save_data(df, database_filename):
    """ 
    parameters: 
    df --> cleaned dataframe
    database_filename --> database filename (w/out extension)
    output:
    <message>
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategories', engine, index=False,if_exists = 'replace')
    return 'saved to database!'


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