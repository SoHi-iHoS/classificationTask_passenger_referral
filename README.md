# classificationTask_passenger_referral

## Project Overview
This project analyzes airline review data spanning from 2006 to 2019 for popular airlines worldwide. The **primary objective** is to predict whether passengers will recommend an airline to their friends based on their experience (binary classification: yes/no).

## Dataset Information
- **Source**: Airline reviews scraped in Spring 2019
- **Original Size**: 131,895 rows × 17 features
- **Time Period**: 2006-2019
- **Content**: Includes multiple choice ratings and free text reviews

## Project Goal

To develop a model that accurately predicts whether a passenger will recommend an airline based on their review and ratings, providing valuable insights for the airline industry on customer satisfaction factors.

## Data Cleaning Process

The original dataset required significant preprocessing:

1. **Basic Cleaning**
   - Removed blank rows (rows with all NaN values)
   - Converted data types as needed for analysis

2. **Date Processing** (`review_date_obj` function)
   - Converted review dates to proper DateTime objects
   - Extracted year information into a separate column
   - Removed the redundant 'date_flown' column

3. **Review Text Cleaning** (`customer_review_transform` function)
   - Applied regex to remove unnecessary patterns:
     - "É?? Trip Verified | ..."
     - "Not Verified | ..."
     - "City to City."

4. **Route Information Extraction** (`create_to_from_via` function)
   - Created three new columns from the 'route' column:
     - 'from' (departure location)
     - 'to' (arrival location)
     - 'via' (connection point, if any)
   - Removed the original 'route' column

5. **Dataset Splitting** (`create_new_df` function)
   - Separated data into:
     - `df_1` (Work Dataset): Contains rows with fewer missing values
     - `df_unseen` (Unseen Dataset): Contains rows with significant missing values (>7)
    
## Data Quality Analysis

### Categorical Variables Validation
A thorough examination was conducted to ensure data integrity:

- **Frequency distribution analysis** for each categorical variable to identify potential anomalies
- **Range verification** to confirm all ordinal variables stayed within defined scales (1-5 rating)
- **Domain validity checks** to ensure categorical values made logical sense within context
- **Cross-referencing** between related variables to identify inconsistencies
- **Historical comparison** against expected distributions based on domain knowledge

No outliers or anomalous entries were found in categorical variables, confirming high data integrity.

### Data Transformation

- **Feature Encoding**:
  - **One-Hot Encoding** for `traveller_type` (with `drop_first=True` to avoid dummy variable trap)
  - **Ordinal Encoding** for `cabin` (Economy: 0, Premium Economy: 1, Business: 2, First Class: 3)
  - Target variable (`recommended`) conversion from 'yes'/'no' to binary 1/0

- **Scaling**:
  - **Min-Max scaling** was chosen because it:
    - Preserves relationships within features
    - Improves interpretability (0-1 range)
    - Handles outliers appropriately
    - Creates feature uniformity across different scales
  - No dimensionality reduction was applied due to the limited feature set (only 10 features)


## Modeling Approaches

Three distinct datasets were created for different modeling approaches:

### 1. Traditional Model (`df_model_one`)
- Applied conventional mean and mode imputations
- Feature breakdown:
  - 2 categorical features (traveller_type & cabin)
  - 7 ordinal features
  - 1 numeric feature
- Imputation strategy:
  - Numeric features: Mean imputation
  - Categorical features: Mode imputation
  - Ordinal features: Mean imputation (values typically around 3 with std dev ~1.3-1.5)

### 2. Probability Mass Function Model (`df_model_pmf`)
- Uses target-based conditional probability to impute missing values
- For each missing value, calculated the most likely value based on:
  - The observed distribution of values for each target class
  - The conditional probability of each value given the target class
- Example process:
  ```
  For missing seat_comfort where recommended=no:
  1. Calculate P(seat_comfort=X|recommended=no) for all X
  2. Assign the most likely value based on these probabilities
  ```

### 3. NLP Sentiment Analysis Model (`df_model_hug`)
- Uses pre-trained Hugging Face transformer for sentiment analysis
- Removed rows with any NaN values
- Limited to reviews under 514 tokens in length for processing efficiency
- Created 4 new features based on sentiment scores:
  - Positive sentiment score (`pos_scores`)
  - Neutral sentiment score (`neu_scores`)
  - Negative sentiment score (`neg_scores`)
  - Final sentiment classification (`senti_final`)
- Specifically uses the `cardiffnlp/twitter-roberta-base-sentiment` model
- Processing time: 25-30 minutes for ~4,500 reviews
- Incorporates text sentiment as predictive features, capturing information not contained in numerical ratings

## Data Preparation Strategies

### Imputation Strategy Impact
The Probability Mass Function (PMF) approach to handling missing values proved most effective because:
- It preserves the relationship between features and the target variable
- Uses conditional probabilities based on the target class
- Provides approximately a 1% improvement across all metrics compared to traditional imputation
- Consistently outperforms traditional imputation regardless of the modeling algorithm used

### Sentiment Analysis Contribution
- The Hugging Face sentiment model features (`df_hug_reg`) achieved the highest F1 score despite limited training data
- This suggests that sentiment extracted from reviews captures crucial aspects of customer satisfaction
- Limited by dataset size (~5,000 reviews) due to processing constraints
- Represents a promising direction for future model enhancement

### Hyperparameter Optimization Findings
- Model improvements came primarily from data preparation strategy rather than hyperparameter tuning
- Random Forest optimization strategy of minimizing train-test accuracy difference resulted in excellent generalization
- LightGBM showed exceptional stability with very low standard deviations across cross-validation metrics# Airline Review Recommendation Prediction

## Model Development

Three machine learning models were developed and optimized:

### 1. Logistic Regression
- **Data Preparation**: Min-Max scaling to normalize features to a 0-1 range
- **Hyperparameter Optimization**: 
  - Grid search with cross-validation
  - Tested various regularization strengths (C values) and penalty types (L1, L2)
  - Used liblinear solver for compatibility with both regularization types
- **Model Interpretation**:
  - SHAP analysis for model-agnostic feature importance
  - Coefficient-based importance ranking
- **Validation**: 5-fold stratified cross-validation

### 2. Random Forest Classifier
- **Initial Setup**: Baseline model with standard parameters
- **Hyperparameter Tuning**:
  - Evaluated combinations of: n_estimators (50, 100, 200), max_depth (2, 5, 7), min_samples_split (2, 5, 10), min_samples_leaf (1, 2, 4)
  - **Optimization strategy**: Minimized the difference between training and testing accuracy to prioritize model generalization over raw performance
- **Feature Analysis**: 
  - Extracted feature importance
  - Created reduced feature set by removing less important features
  - Tested model performance on reduced feature set

### 3. LightGBM Classifier
- **Hyperparameter Optimization**:
  - GridSearchCV with 3-fold cross-validation
  - Parameter grid explored: num_leaves, max_depth, learning_rate, n_estimators, and min_child_samples
  - Accuracy as the optimization metric
- **Model Evaluation**:
  - 5-fold stratified cross-validation
  - Multiple metrics: accuracy, F1 score, ROC-AUC, precision, and recall
- **Results**:
  - Exceptional stability across different data samples (low standard deviations)
  - Near-perfect ROC-AUC score of 0.9922
  - Well-balanced precision and recall

## Key Findings

### Model Performance Comparison
All models performed exceptionally well (>95% on all metrics), but with notable differences:

- **Best Overall Performance**: The PMF-based imputation strategy consistently provided superior results across all models
- **Most Predictive Power**: Both Logistic Regression and LightGBM with PMF-imputed data achieved the highest overall metrics
- **Best F1 Score**: The Logistic Regression model with Hugging Face dataset (`df_hug_reg`) achieved the highest F1 score (0.975677) despite being trained on only ~5,000 data points
- **Balanced Performance**: The PMF model without sentiment features (`df_pmf_reg`) provided the best overall balance of performance metrics

### Final Model Selection
**Logistic Regression with PMF-imputed data** was selected as the final production model because:
- Excellent performance with high accuracy and recall
- Slightly better ROC AUC score compared to LightGBM
- More time and cost-efficient to train compared to ensemble models
- Better interpretability for business stakeholders

### Feature Importance
SHAP analysis identified the following as the most important predictors:
- `overall` rating
- `ground_service`
- `value_for_money`
- `cabin_service`
- `entertainment`
- `food_bev`

These key features were selected for the final production model.

## Business Impact Analysis

### Evaluation Metrics Selection
For this project, our focus was on metrics that translate to positive business outcomes:

- **Recall**: Prioritized as it represents the ability to identify passengers who would recommend the airline (>98.5% across models)
  - Failing to identify potential promoters represents a missed business opportunity
  - Nearly 99.5% of passengers who would recommend the airline can be identified by the top model

- **Precision**: Important for ensuring efficient resource allocation (96.77% for the best model)
  - Ensures marketing resources are directed toward true promoters
  - Minimizes wasteful investments in passengers unlikely to recommend

- **F1 Score**: Balanced metric that captures both precision and recall performance
  - Particularly strong in sentiment-enhanced models

### Business Implications

1. **Targeted Marketing Opportunities**:
   - The model can identify nearly 99.5% of passengers who would recommend the airline
   - These identified promoters can be leveraged for testimonials, referral programs, and advocacy initiatives

2. **Resource Optimization**:
   - High precision ensures resources are efficiently allocated
   - Marketing campaigns can focus on actual promoters with confidence

3. **Feature Engineering Value**:
   - The superior performance of PMF-based models demonstrates the value of transforming raw features into probability distributions
   - This approach better represents passenger behavior patterns

4. **Future Potential**:
   - Combined approach using sentiment analysis with PMF imputation represents a promising direction
   - Resource constraints limited full exploration of this hybrid approach
