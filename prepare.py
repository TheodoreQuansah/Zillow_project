import pandas as pd

from math import sqrt
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer





def eval_model(y_actual, y_hat):
    """
    Calculate and return the root mean squared error (RMSE) between actual and predicted values.

    Parameters:
    y_actual (array-like): The actual target values.
    y_hat (array-like): The predicted target values.

    Returns:
    float: The calculated root mean squared error (RMSE) rounded to two decimal places.
    """
    return round(sqrt(mean_squared_error(y_actual, y_hat)), 2)

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train a machine learning model, make predictions on training and validation sets, and print RMSE for both sets.

    Parameters:
    model: The machine learning model to be trained.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target variable.

    Returns:
    Trained machine learning model.
    """
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {round(train_rmse, 2)}.')
    print(f'The validate RMSE is {round(val_rmse, 2)}.')
    
    return model

def convert_and_rename_dummies(train, val, test):
    """
    Convert specified columns into dummies and rename them.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with dummies and renamed columns.
    """
    # Specify columns to convert into dummies
    columns_to_convert = ['county', 'state']

    # Perform conversion
    train = pd.get_dummies(train, columns=columns_to_convert)
    val = pd.get_dummies(val, columns=columns_to_convert)
    test = pd.get_dummies(test, columns=columns_to_convert)

    # Rename columns
    column_mapping = {
        'county_Los Angeles County': 'los_angeles_county',
        'county_Orange County': 'orange_county',
        'county_Ventura County': 'ventura_county',
        'state_CA': 'california'  # Corrected typo 'carlifornia' to 'california'
    }
    train = train.rename(columns=column_mapping)
    val = val.rename(columns=column_mapping)
    test = test.rename(columns=column_mapping)

    return train, val, test

def scaled_data(train, val, test, scaler_type='standard'):
    """
    Scale numerical features in train, val, and test datasets using various scaling techniques.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.
    scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust', 'quantile').

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with scaled numerical features.
    """
    t_to_scale = to_scale = train.drop(columns=['tax_value']).columns
    v_to_scale = to_scale = val.drop(columns=['tax_value']).columns
    te_to_scale = to_scale = test.drop(columns=['tax_value']).columns
    
    # Initialize the selected scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', 'robust', 'quantile'.")

    # Fit the scaler on the training data and transform all sets
    train[t_to_scale] = scaler.fit_transform(train[t_to_scale])
    val[v_to_scale] = scaler.transform(val[v_to_scale])
    test[te_to_scale] = scaler.transform(test[te_to_scale])

    return train, val, test

def xy_split(df):
    """
    Split a DataFrame into features (X) and the target variable (y) by dropping the 'tax_value' column.

    Parameters:
    df (pd.DataFrame): DataFrame to be split.

    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    return df.drop(columns=['tax_value']), df.tax_value




