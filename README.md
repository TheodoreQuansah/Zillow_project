#  Estimating Home Value: A Zillow Data Science Project
 
# Project Description
 
As a junior data scientist on the Zillow data science team, my mission is to develop an accurate and insightful regression model to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had transactions during the year 2017. This project aims to improve an existing model and provide valuable insights into the key drivers of property value for single-family homes.


# Project Scenario:

I have received a directive from the Zillow Data Science Team to enhance their existing property value prediction model. The team is open to a wide range of strategies, from creating new features from existing data to experimenting with non-linear regression algorithms. My fresh perspective as a newcomer to the team is highly valued. Additionally, there is a slight twist: the locations of these properties have been lost due to an email mishap. To accurately assess property taxes, I need to determine which states and counties these properties are situated in.


# Project Goal
 
* Regression Model Development: Your primary goal is to construct a robust Machine Learning Regression model that accurately predicts the property tax assessed values of Single Family Properties. You will leverage various property attributes as input features to achieve this.
* Key Drivers of Property Value: Investigate and uncover the critical factors influencing property values for single-family homes. This analysis will address questions such as why seemingly similar properties have different values based on their physical attributes and locations.
* Data-Driven Insights: Provide the Zillow Data Science Team with a comprehensive report detailing the steps taken during the project, the rationale behind each decision, and the outcomes achieved. This report should be structured in a way that allows the team to replicate your work and gain a deep understanding of the project's methodology.
* Recommendations: Offer well-founded recommendations regarding what strategies and techniques work best for predicting property values, and equally important, what doesn't work. Your insights will inform future decision-making and model enhancements.
 

# Initial Questions

* Does squarefeet affect tax value?
* Does FIPS affect tax value?  
* Is there a correlation between tax value and number of bedrooms? 
* Does the year the property was built affect the value?   
* Is there a relationship between tax value and number of bathrooms?

# Initial Thoughts
 
* The target variable, 'taxvaluedollarcnt,' is likely influenced by various factors, such as property size, location, and condition. Investigating these factors' impact will be crucial.
* Property tax assessments at the county level suggest that regional factors may play a significant role in property values. Examining the variations among different counties will be important.
* Creating new features or engineering existing ones, like price per square foot or property age, may improve the model's predictive accuracy.
* Since we are dealing with property values, there might be non-linear relationships between features and the target variable. Exploring non-linear regression algorithms could be beneficial.
* Addressing missing location information is essential for geospatial analysis, which can provide insights into property values based on states and counties.
* I hypothesize that attributes such as square footage, number of bedrooms, and bathrooms will be strong drivers of property values, but their impact might vary by location.
* It will be interesting to identify any unexpected factors that influence property values, as this can lead to valuable insights and model improvements.

 
# The Plan
 
* - Acquire
    - Acquiring specific columns that i need for the project from the Zillow dataset using sql.
    - Read the sql query into a dataframe.
 
* Prepare data
   * Create new columns by transforming and utilizing existing data features.
       * Check all rows in the df with no/null values and dropping them.
       * Use MinMax scaler on the numerical columns.
       * Group different values in the columns and make them categorical if needed.
       * One-hot encode categorical culomns with get dummies.
       * Drop all the encoded columns that are not useful.
      
 
* Explore data in search of drivers of churn
   * Answer the following initial questions
        * Does squarefeet affect tax value?
        * Does FIPS affect tax value?  
        * Is there a correlation between tax value and number of bedrooms? 
        * Does the year the property was built affect the value?   
        * Is there a relationship between tax value and number of bathrooms?
      
* Create 3 predictive models to closely auspicate the prices of houses.
   * Utilize insights from exploratory analysis to construct predictive models of various types.
   * Assess model performance using both training and validation datasets.
   * Choose the optimal model by considering the highest accuracy.
   * Assess the chosen top-performing model using the test dataset.
 
* Draw conclusions
 
# Data Dictionary

  * This table provides a clear definition of each feature present in the dataset along with their respective descriptions.


| Feature                | Definition | Data Type |
|:-----------------------|:-----------|:----------|
| finishedsquarefeet12   | Finished square footage of the property | float64 |
| squarefeet             | Square footage of the property | float64 |
| latitude               | Latitude coordinate of the property | float64 |
| regionidzip            | Region ID of the property's ZIP code | float64 |
| longitude              | Longitude coordinate of the property | float64 |
| lotsizesquarefeet      | Square footage of the property's lot | float64 |
| logerror               | Logarithm of the error in the property's Zillow estimate | float64 |
| year_built             | Year the property was built | float64 |
| id                     | Identifier for the property | int64 |
| rawcensustractandblock | Raw census tract and block identifier | float64 |
| regionidcity           | Region ID of the property's city | float64 |
| bathrooms              | Number of bathrooms in the property | float64 |
| bedrooms               | Number of bedrooms in the property | float64 |
| fips                   | Federal Information Processing Standards (FIPS) code for the property's location | float64 |
| tax_value              | Property tax value | float64 |
| bedrooms_bin           | Categorical variable representing a bin for the number of bedrooms | category |
| bathrooms_bin          | Categorical variable representing a bin for the number of bathrooms | category |
| squarefeet_bin         | Categorical variable representing a bin for the square footage of the property | category |
| decades                | Categorical variable representing the decade in which the property was built | category |


 
# Steps to Reproduce
1) Clone this repository by clicking on the SSH link.
2) If you have been granted access to the Codeup MySQL database(i.e. ask for permission from staff if you do not have access):
   i) Save the env.py file in the repository. Make sure to include your user, password, and host variables.
   ii) Add the env.py file to the .gitignore file to keep your sensitive information secure.
4) Run the Jupyter Notebook file in your local environment.
 ............................................................................................................................................................................
# Conclusions
- Having tech support greatly reducing the customers probability of churning.
- The longer the customer has been with the company the less likely they are to churn.
- New customers tend to have a high churn rate.
- Customers with a monthly plan have a higher churn rate than customers with a yearly plan.
- Customers with DSL tend to have a significantly lower churn rate compared to customers with Fiber optics.
- It is apparent that senior citizens, although a minority of customer churn, exhibit a considerably higher churn rate in comparison to non-senior citizen customers.

# Takeaways
i) Contract Type: In the decision tree model, contract type stands out as the most crucial feature with an importance score of around 0.619. It also holds significance in the random forest model, ranking second with an importance score of approximately 0.1987.

ii) Tenure: Tenure is the second most important feature in the decision tree model, with an importance score of roughly 0.194. Interestingly, it takes the lead in importance in the random forest model, where its score is approximately 0.209.

iii) Monthly Charges: Monthly charges emerge as the third most important feature in the random forest model, with an importance score of about 0.142.

iv) Internet Service Type: In the decision tree model, internet service type holds the third position in importance, with a score of roughly 0.167. It also maintains significance in the random forest model, ranking fifth with an importance score of approximately 0.113.

v) Total Charges: Total charges play a role in churn prediction and are considered the fourth most important feature in the random forest model, with an importance score of approximately 0.132.

 
# Recommendations
- Encourage customers to choose longer-term contracts, such as one or two-year contracts, as these seem to have a positive impact on customer retention.
- Consider implementing retention strategies or loyalty programs that reward customers for their longevity with the company.
- Explore options to optimize pricing strategies to make them more competitive in the market without compromising quality.
- Continuously collect feedback from customers who have churned to understand their reasons for leaving. Use this information to make improvements in areas that matter most to customer