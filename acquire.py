import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from env import get_connection
from sklearn.model_selection import train_test_split


def get_properties_2017():
    """
    Retrieve properties data for the year 2017.

    Returns:
    pd.DataFrame: DataFrame containing properties data for 2017.
    """
    # Define the filename for the CSV file
    filename = 'properties_2017mock3.csv'
    
    # Check if the CSV file already exists
    if os.path.isfile(filename):
        # If the file exists, read it into a DataFrame and return it
        return pd.read_csv(filename)
    else:
        # If the file doesn't exist, define an SQL query to retrieve data
        query = '''
                SELECT  finishedsquarefeet12, 
                        calculatedfinishedsquarefeet,
                        latitude,
                        regionidzip, 
                        longitude,
                        lotsizesquarefeet, 
                        logerror, 
                        yearbuilt,
                        properties_2017.id,
                        predictions_2017.id,
                        rawcensustractandblock,
                        regionidcity,
                        properties_2017.parcelid,
                        predictions_2017.parcelid,
                        bathroomcnt,      
                        bedroomcnt,
                        fips,
                        taxvaluedollarcnt
                FROM properties_2017
                LEFT JOIN predictions_2017 ON predictions_2017.parcelid = properties_2017.parcelid
                WHERE EXTRACT(YEAR FROM predictions_2017.transactiondate) = 2017
                AND propertylandusetypeid = 261;
                '''
        
        # Get a connection URL (you may want to define the 'get_connection' function)
        url = get_connection('zillow')  # You'll need to define this function
        
        # Execute the SQL query and read the result into a DataFrame
        df = pd.read_sql(query, url)
        
        # Save the result to a CSV file
        df.to_csv(filename, index=False)

        # Return the DataFrame
        return df

def clean_and_converts():
    """
    Clean and convert columns in the properties DataFrame.

    Returns:
    pd.DataFrame: Cleaned and converted DataFrame.
    """
    # Replace this line with how you obtain your DataFrame
    df = get_properties_2017()

    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'squarefeet',
        'taxamount' : 'tax_amount',
        'taxvaluedollarcnt': 'tax_value',
        'yearbuilt': 'year_built'
    })

    bed_edges = [0, 2, 3, 4, 6, 8, 11, 25]
    bath_edges = [0, 1, 2, 3, 4, 7, 15, 32]
    sf_edges = [1, 500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000]
    dec_edges = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

    # Apply cut and assign the results to respective columns
    df['bedrooms_bin'] = pd.cut(df['bedrooms'], bins=bed_edges, right=False)
    df['bathrooms_bin'] = pd.cut(df['bathrooms'], bins=bath_edges, right=False)
    df['squarefeet_bin'] = pd.cut(df['squarefeet'], bins=sf_edges, right=False)
    df['decades'] = pd.cut(df['year_built'], bins=dec_edges, right=False)
    
    df['bedrooms_bin'] = df['bedrooms_bin'].apply(lambda x: x.right)
    df['bathrooms_bin'] = df['bathrooms_bin'].apply(lambda x: x.right)
    df['squarefeet_bin'] = df['squarefeet_bin'].apply(lambda x: x.right)
    df['decades'] = df['decades'].apply(lambda x: x.right)

    # Drop rows with any null values
    df = df.dropna()

    df['state'] = 'CA'

    # Define a dictionary to map FIPS codes to county names
    fips_to_county = {
    6037: 'Los Angeles County',
    6059: 'Orange County',
    6111: 'Ventura County'
    }

    # Create the 'county' column by mapping the FIPS codes
    df['county'] = df['fips'].replace(fips_to_county)

    # Convert all columns to integers
    # df = df.astype(int)

    return df

def train_val_test(df):
    """
    Split the DataFrame into training, validation, and test sets.

    Returns:
    pd.DataFrame: Training, validation, and test DataFrames.
    """
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    # Return the three datasets
    return train, val, test

def compare_data(scaled_col, x_lim, df, original='tax_value'):
    """
    Compare histograms of original and scaled data.

    Parameters:
    scaled_col (str): The name of the scaled column to compare.
    x_lim (int): The x-axis limit for the histograms.
    df (pd.DataFrame): The DataFrame containing the data.
    original (str): The name of the original data column.

    Returns:
    None
    """
    # Create a figure with two side-by-side subplots for comparison
    plt.figure(figsize=(11, 7))
    
    # Left Subplot: Original Data Histogram
    plt.subplot(1, 2, 1)
    
    # Create a histogram of the original data (original column)
    sns.histplot(data=df, x=original, bins=20)
    
    # Set x-axis limits for the original data histogram
    plt.xlim(0, 90_000_000)
    
    # Set y-axis limits for the original data histogram
    plt.ylim(0, 100)
    
    # Right Subplot: Scaled Data Histogram
    plt.subplot(1, 2, 2)
    
    # Create a histogram of the scaled data (scaled_col)
    sns.histplot(data=df, x=scaled_col, bins=20)
    
    # Set x-axis limits for the scaled data histogram
    plt.xlim(0, x_lim)
    
    # Set y-axis limits for the scaled data histogram
    plt.ylim(0, 100)
    
    # Display the side-by-side histograms
    plt.show()











