# classificationTask_passenger_referral

## Project Overview
This project analyzes airline review data spanning from 2006 to 2019 for popular airlines worldwide. The **primary objective** is to predict whether passengers will recommend an airline to their friends based on their experience (binary classification: yes/no).

## Dataset Information
- **Source**: Airline reviews scraped in Spring 2019
- **Original Size**: 131,895 rows × 17 features
- **Time Period**: 2006-2019
- **Content**: Includes multiple choice ratings and free text reviews

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
- Created 3 new features based on sentiment scores:
  - Positive sentiment score
  - Neutral sentiment score
  - Negative sentiment score
- Incorporates text sentiment as predictive features, capturing information not contained in numerical ratings

## Project Goal

The goal is to develop a model that accurately predicts whether a passenger will recommend an airline based on their review and ratings, providing valuable insights for the airline industry on customer satisfaction factors.

## Usage

[Add instructions on how to use the code/models here]

## Requirements

[Add required libraries and dependencies here]

## License

[Add license information here]
