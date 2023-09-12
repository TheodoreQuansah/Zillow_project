import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def pearson_correlation(x, y, alpha=0.05):
    """
    Calculate Pearson's correlation coefficient (r) between two arrays or lists of data and perform hypothesis testing.

    Parameters:
    x (array-like): The first data array.
    y (array-like): The second data array.
    alpha (float): The significance level for hypothesis testing (default is 0.05).

    Returns:
    str: A result message based on hypothesis testing.
    float: Pearson's correlation coefficient (r) between x and y.
    float: p-value associated with the correlation coefficient.

    Note:
    - The function returns both the correlation coefficient (r) and the p-value.
    - The p-value indicates the statistical significance of the correlation.
    - If p-value is less than alpha, you can reject the null hypothesis of no correlation.
    """
    correlation_coefficient, p_value = stats.pearsonr(x, y)
    
    if p_value < alpha:
        result = "We reject the null hypothesis. There appears to be a relationship."
    else:
        result = "We fail to reject the null hypothesis."
    
    return result, (f'r = {correlation_coefficient}.'), (f'p = {p_value}.')

def perform_anova(df, alpha=0.05):
    """
    Perform ANOVA on tax_value by FIPS regions.

    Parameters:
    - df: DataFrame containing 'tax_value' and 'fips' columns.
    - alpha: Significance level for hypothesis testing (default is 0.05).

    Returns:
    - A string indicating the result of the ANOVA test.
    """
    # Perform ANOVA
    anova_result = stats.f_oneway(
        df[df['fips'] == 6037]['tax_value'],
        df[df['fips'] == 6059]['tax_value'],
        df[df['fips'] == 6111]['tax_value']
    )

    if anova_result.pvalue < alpha:
        result = "We reject the null hypothesis. There is a significant difference among FIPS regions."
    else:
        result = "We fail to reject the null hypothesis."

    return result, anova_result.pvalue

def pearson_correlation_test(data1, data2):
    """
    Perform a Pearson correlation test between two datasets.

    Parameters:
    data1 (pd.Series): First dataset for correlation analysis.
    data2 (pd.Series): Second dataset for correlation analysis.

    Returns:
    correlation_coefficient (float): Pearson correlation coefficient.
    p_value (float): Two-tailed p-value.
    """

    alpha = 0.05
    correlation_coefficient, p_value = stats.pearsonr(data1, data2)

    if p_value < alpha:
        result = "We reject the null hypothesis. There is a correlation between tax value and number of bedrooms."
    else:
        result = "We fail to reject the null hypothesis."
    
    return result, correlation_coefficient, p_value

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
    Train a machine learning model and evaluate its performance on training and validation data.

    Parameters:
    model: The machine learning model to be trained.
    X_train (array-like): Training features.
    y_train (array-like): Training target values.
    X_val (array-like): Validation features.
    y_val (array-like): Validation target values.

    Returns:
    model: Trained machine learning model.
    """
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {round(train_rmse, 2)}.')
    print(f'The validate RMSE is {round(val_rmse, 2)}.')
    
    return model

def test_model(model, X_train, y_train, X_test, y_test):
    """
    Train a machine learning model and evaluate its performance on training and validation data.

    Parameters:
    model: The machine learning model to be trained.
    X_train (array-like): Training features.
    y_train (array-like): Training target values.
    X_test (array-like): Test features.
    y_test (array-like): Test target values.

    Returns:
    model: Trained machine learning model.
    """
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    test_preds = model.predict(X_test)
    
    test_rmse = eval_model(y_test, test_preds)
    
    print(f'The train RMSE is {round(train_rmse, 2)}.')
    print(f'The test RMSE is {round(test_rmse, 2)}.')
    
    return model

def different_plots(df, categorical_col, continuous_col, plot):
    """
    Generate different types of plots to visualize relationships between categorical and continuous variables.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    categorical_col (str): The name of the categorical column.
    continuous_col (str): The name of the continuous column.
    plot (str): Type of plot to generate ('scatter', 'box', 'swarm', 'bar', 'hist').

    Returns:
    None
    """
    # Set the color to blue for all plots
    color = 'turquoise'

    if plot == "scatter":
        # Scatter plot to visualize the relationship between the categorical and continuous variables
        plt.figure(figsize=(8, 4))
        sns.scatterplot(data=df, x=categorical_col, y=continuous_col, color=color)
        plt.title(f'Distribution of {continuous_col} by {categorical_col}')
        plt.xticks(rotation=45)
        plt.xlabel(categorical_col)
        plt.ylabel(continuous_col)
        plt.show()

    elif plot == "box":
        # Box plot to visualize the distribution of the continuous variable by category
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=categorical_col, y=continuous_col, color=color)
        plt.title(f'Distribution of {continuous_col} by {categorical_col}')
        plt.xticks(rotation=45)
        plt.xlabel(categorical_col)
        plt.ylabel(continuous_col)
        plt.show()

    elif plot == "swarm":
        # Swarm plot to visualize the distribution of the continuous variable by category
        plt.figure(figsize=(8, 4))
        sns.swarmplot(data=df, x=categorical_col, y=continuous_col, color=color)
        plt.title(f'Distribution of {continuous_col} by {categorical_col}')
        plt.xticks(rotation=45)
        plt.xlabel(categorical_col)
        plt.ylabel(continuous_col)
        plt.show()

    elif plot == "bar":
        # Bar plot to visualize the mean of the continuous variable by category
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x=categorical_col, y=continuous_col, color=color)
        plt.title(f'{continuous_col} by {categorical_col}')
        plt.xticks(rotation=45)
        plt.xlabel(categorical_col)
        plt.ylabel(f'{continuous_col}')
        plt.show()

    elif plot == "hist":
        # Bar plot to visualize the mean of the continuous variable by category
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=categorical_col, y=continuous_col, color=color)
        plt.title(f'{continuous_col} by {categorical_col}')
        plt.xticks(rotation=45)
        plt.xlabel(categorical_col)
        plt.ylabel(f'{continuous_col}')
        plt.show()
    
    else:
        print("Invalid plot type. Supported types: 'scatter', 'box', 'swarm', 'bar', 'hist'")


def polynomial_feature_expansion(X_train, X_val, X_test, degree=2):
    """
    Perform polynomial feature expansion for training and validation datasets.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_val (pd.DataFrame): Validation features.
    X_test (pd.DataFrame): Test features.
    degree (int): Degree of polynomial features to be created (default is 2).

    Returns:
    pd.DataFrame: Training features with polynomial expansion.
    pd.DataFrame: Validation features with polynomial expansion.
    pd.DataFrame: Test features with polynomial expansion.
    """
    # Create an instance of PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)

    # Transform the training features into polynomial features
    X_train_s = poly.fit_transform(X_train)

    # Transform the validation features into polynomial features
    X_val_s = poly.fit_transform(X_val)

    # Transform the test features into polynomial features
    X_test_s = poly.fit_transform(X_test)

    return X_train_s, X_val_s, X_test_s, poly

def create_baselines(y_train):
    """
    Create a DataFrame 'baselines' with columns 'y_actual,' 'y_mean,' and 'y_median.'

    Parameters:
    y_train (pd.Series or array-like): Actual target values from the training dataset.

    Returns:
    pd.DataFrame: DataFrame containing 'y_actual,' 'y_mean,' and 'y_median' columns.
    """
    baselines = pd.DataFrame({
        'y_actual': y_train,
        'y_mean': y_train.mean(),
        'y_median': y_train.median()
    })
    
    return baselines


def plot_rmse_bar_chart():
    """
    Generate a bar chart with RMSE values displayed on top of the bars.

    Parameters:
    rmse_values (list): List of RMSE values for each dataset.
    datasets (list): List of dataset labels.
    title (str): Title of the chart.

    Returns:
    None
    """

    rmse_values = [245941.25, 355484.24, 356434.56]
    datasets = ['Train', 'Validate', 'Test']
    title = 'Model Performance on Different Datasets (RMSE)'
    # Create a bar chart to visualize RMSE
    plt.figure(figsize=(8, 4))
    bars = plt.bar(datasets, rmse_values, color='turquoise')

    plt.xlabel('Dataset')
    plt.ylabel('RMSE Value')
    plt.title(title)

    # Add RMSE values on top of the bars
    for bar, rmse in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 1000, f'{rmse:.2f}', ha='center', color='black', fontsize=10)

    plt.show()

