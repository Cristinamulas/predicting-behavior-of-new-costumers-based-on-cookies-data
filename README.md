# Predicting-behavior-of-new-costumers-based-on-cookies-data

## GOAL

The goal of this project is to build classifiers to predict the behavior of a future purchase based on cookie history and identificate which variables are most informative of purchase.

## DATASET DESCRIPTION

The dataset consists of the history of users who bought a product or those who didn't buy the product in a period of interest. The dataset contains 54,584 records and 14 attributes.

Some of the characteristics of this dataset are:

- is_buyer: a feature that is a binary variable encoded as 0 or 1, similar to the target feature. The rest of the records are quantitative values.
- buy_freq: has many null values.
- last_buy and last_visit: contain very similar information.
- uniq_urls: has -1 as minimum value. This can be a placeholder.
- num_checkins: maximum value is too high, it can be an outlier.
- y_buy: a feature that is the target value and it is imbalanced. 

## TARGET FEATURE

The target feature is a binary variable encoded as 0 if they didn't purchase the product in a period of interest and 1 if they did purchase the product in a period of interest. The target feature is highly imbalanced, it has a higher % of users that didn't buy the product. Exactly, 99.52% of users didn't buy the product and only 0.48% of users bought the product. 

<img width="494" alt="Screen Shot 2021-03-04 at 4 06 03 PM" src="https://user-images.githubusercontent.com/43222117/110340238-0b68cf80-7ff7-11eb-9b51-1db4c4b4c39d.png">
