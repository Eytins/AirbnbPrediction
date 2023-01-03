[0:id, 7:neighborhood_overview, 9:host_id, 15:host_response_time, 16:host_response_rate, 17:host_acceptance_rate(Survivorship bias?), 18:host_is_superhost, 26:host_identity_verified, 28:neighbourhood_cleansed, 30:latitude, 31:longitude, 32:property_type, 33:room_type, 34:accommodates, 36:bathrooms_text, 37:bedrooms, 38:beds, 39:amenities, 40:price, 41:minimum_nights, 50:has_availability, 51:availability_30, 56:number_of_reviews, (57:number_of_reviews_ltm, 58:number_of_reviews_l30d,) 59:first_review, 69:instant_bookable, 70:calculated_host_listings_count, 74:reviews_per_month]

# Idea
Fill the blanks: use scores to train a model for target, then use this model to fill the blanks.

cut:
[](](]


## Fixed: property_type / bathrooms_text
Amount less than 100 replace to others, or not?

## Fixed: amenities
The same procedure with reviews



Machine Learning Final Assignment
Mingzhe Liu | 22306186

Predict review_scores_rating
Review_scores_rating is calculated as a weighted sum of other scores. So I'm going to predict this feature first, as it will be simpler and more accurate, and I can build a template for the model in the process of predicting this feature, so that I can focus more on other steps such as feature selection and data pre-processing in the process of predicting other values.
Data preprocessing
As described above, in predicting this value I chose six other ratings as feature values, the original data contained some null values in these columns, I removed these null values and the amount of data changed from 7566 to 6078. Before training the model, to check for abnormal values, I also checked the maximum and minimum values of all the data and got 0 and 5 respectively, which clearly met the requirements. In addition to this, I also plotted the frequency distribution of the original data, from which I could see that most of the ratings were distributed between 4.5 and 5.0.

Train and validation
For the prediction of this value, I chose to train each of the following models, Linear Regression, Lasso and Ridge models.
Linear Regression
I trained a simple Linear Regression model with all default parameters, and the results were as follows:
Slopes: [0.11666305 0.28772331 0.04143592 0.18858207 0.08296101 0.32111273]
Intercept: -0.1892600757420686
Accuracy: 0.8213732138917692
The formula used in the prediction is
y=0+1x
0 is the intercept, 1 is the slopes. Enter x to get the predicted y. The loss function of this model is
J(0,1)=1mi=1m(h(x(i))-y(i))2
The model will be trained by selecting the appropriate (0,1) to make the above equation as small as possible. This model yields an accuracy of around 80%. The scoring mechanism measures the difference between all predicted values and the mean of the true values, using the difference between all predicted values and the individual true values as a benchmark.
In this process, I used DummyRegressor as the baseline model and obtained a score:
Baseline score: -0.08559828782863588
It follows that this Linear Regression is a well-performing model.
The R^2 score is negative, which means that the model is worse at predicting the responses than a simple mean model that always predicts the mean of the observed responses. This can occur if the model has a large MSE or if the variance of the observed responses is very small.
Lasso
Compared to the Linear regression above, Lasso regression is a type of linear regression that uses a regularisation known as the L1 penalty, to reduce the complexity of the model by shrinking the coefficients of the features towards zero. The L1 penalty encourages sparsity in the model, which means that it is more likely to set the coefficients of unimportant features to zero, effectively removing them from the model.

Ridge

