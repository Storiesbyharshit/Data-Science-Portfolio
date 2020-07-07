In this we are having a dataset of a car company which we will use to anayse the current trends in markets and help a company to understand the significant 
price influencing trends in the industry .
We will be using the statsmodel library for modelling .

We will follow the given steps :
> * Reading and Understanding the Data
> * Data Cleaning and Preparation
> * Visualizing the data and EDA
> * List of significant variables after Visual analysis
> * Train-Test Split and feature scaling
> * Feature Selection
> * Modelling various statsmodel and comparing results
> * Final Inference


### Modelling Goal
You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly 
the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels.
Further, the model will be a good way for management to understand the pricing dynamics of a new market.

### Goals 
>- Which variables are significant in predicting the price of a car
>- How well those variables describe the price of a car

### Inference 

> * **R-sqaured and Adjusted R-squared (extent of fit)**- 0.899 and 0.896 - 90% variance explained.

> * **F-stats and Prob(F-stats) (overall model fit)** - 308.0 and 1.04e-67(approx. 0.0) - Model fit is significant and explained 90% variance is just not by chance.

> * **p-values** - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the predictors are statistically significant.
