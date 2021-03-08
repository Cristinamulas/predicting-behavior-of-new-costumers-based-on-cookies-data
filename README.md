# Predicting Behavior Of New Costumers Based On Cookies Data

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

## EXPLORATORY DATA ANALYSIS

Questions that I answered while exploring the database:

### How many users bought the product in the past?

<img width="494" alt="Screen Shot 2021-03-04 at 4 26 48 PM" src="https://user-images.githubusercontent.com/43222117/110340417-3e12c800-7ff7-11eb-9647-b13338110455.png">

- The majority of the users didn't buy the product in the past.
### Who visits the website more often, new or old users?


<img width="826" alt="Screen Shot 2021-03-05 at 4 30 16 PM" src="https://user-images.githubusercontent.com/43222117/110340496-57b40f80-7ff7-11eb-91d1-5356daeab6aa.png">

- Buyers that purchased the product in the past visited the website more often.

### Which users browse around the website more often, new or old users?

<img width="842" alt="Screen Shot 2021-03-05 at 7 33 36 PM" src="https://user-images.githubusercontent.com/43222117/110340645-79ad9200-7ff7-11eb-90ea-c7e3e3f50863.png">

- Users who didn't buy the product in the past look at more urls when they are browsing the website.

## MODEL

In this project, I started with the implementation of Logistic Regression as a baseline model. Logistic Regression is a simple algorithm and it has a reasonable chance of providing good results. Afterwards, I built an ensemble method because I knew that it would improve my results by combining several models. Ensambles methods tend to have a better predictive performance compared to a single model. The ensemble that I chose was XGBoost. Some of the advantages XGBoost are that its very easy to implement, its computationally efficient, it uses 

regularized boosting that prevents overfitting, it can handle missing values automatically, it can be run in parallel, it can cross validate at each interaction and finally, for tree pruning, XGboost by default will go very deep and then tries to prune that tree backwards so that results are generally in deeper trees but more highly optimized trees.

I knew that the accuracy of these models could improve so I tuned the parameters for each classifier using a combinatorial grid search. For this task, I used the GridSearchCV method that finds the best combination of parameters for a given model. Grid search is sometimes referred to as an exhaustive search because it tries every single combination and as a consequence it's computationally expensive. I implemented this method within 4 cross validation rounds.

## RESULTS

### Logistic Refression

<img width="484" alt="Screen Shot 2021-03-05 at 3 25 31 PM" src="https://user-images.githubusercontent.com/43222117/110341161-0bb59a80-7ff8-11eb-9722-2d86d1893523.png">

<img width="623" alt="Screen Shot 2021-03-05 at 3 35 58 PM" src="https://user-images.githubusercontent.com/43222117/110341177-1112e500-7ff8-11eb-9964-4481e7f9a3f9.png">

### The most important features of Logistic Regression
<img width="919" alt="Screen Shot 2021-03-05 at 3 36 16 PM" src="https://user-images.githubusercontent.com/43222117/110341254-2425b500-7ff8-11eb-80c0-6e890058109e.png">

### XGBoost
<img width="468" alt="Screen Shot 2021-03-05 at 3 50 49 PM" src="https://user-images.githubusercontent.com/43222117/110341467-5afbcb00-7ff8-11eb-85b9-2c44aa799829.png">

<img width="662" alt="Screen Shot 2021-03-05 at 3 51 09 PM" src="https://user-images.githubusercontent.com/43222117/110341474-5d5e2500-7ff8-11eb-93e2-339fac2dbe7e.png">

### The most important features of XGBoost





