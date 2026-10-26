Predicting Hotel Booking Cancellations

A Python project that uses machine learning to predict whether a hotel booking will be cancelled, and to explore which booking characteristics are most closely associated with cancellation.

Aim

Can we predict whether a hotel booking will be cancelled, and which factors are most associated with cancellations?

Data

The project uses the Hotel Booking Demand dataset from Kaggle. It contains 119,390 bookings from a city hotel and a resort hotel, described by 32 columns such as lead time, price, deposit type, customer type and country.

The data is not included in this repository. To run the notebook, download hotel_bookings.csv from the link above and save it in the same folder.

Approach

Cleaning. I removed four columns. reservation_status and reservation_status_date record what ultimately happened to each booking, so keeping them would have given the model the answer (data leakage). agent and company were either mostly missing or simply ID codes.

I also found that 32,280 rows (27%) were exact duplicates once those columns were removed. I dropped them, leaving 87,110 bookings, of which 27% were cancelled. This made a noticeable difference to the results (see below).

Exploration. I looked at cancellation rates by hotel, deposit type, customer type, lead time and number of special requests.

Modelling. I split the bookings into a training set (80%) and a test set (20%). The split was stratified, so both sets kept the same cancellation rate, and I fixed the random seed so the results can be reproduced. Preprocessing was built as a scikit-learn pipeline and fitted on the training data only: missing numeric values were filled with the median and scaled, missing categorical values were filled with the most common value, and categories were one-hot encoded (categories with fewer than 100 bookings were grouped together).

I compared two models:

Logistic Regression, as a simple baseline.
Random Forest with 200 trees, to test whether a more flexible model would perform better.

Both were evaluated on the test set using ROC-AUC, precision, recall and a confusion matrix. Finally, I looked at the Random Forest's feature importances, adding together the one-hot columns that came from the same original feature.

Results

Scores on the test set (17,422 bookings):

Model	ROC-AUC	Accuracy	Precision (cancelled)	Recall (cancelled)
Logistic Regression	0.843	0.79	0.68	0.47
Random Forest	0.898	0.84	0.76	0.62

The Random Forest identified 62% of the bookings that were genuinely cancelled, compared with 47% for the baseline, and 76% of its cancellation predictions were correct. Its confusion matrix was:

Predicted kept	Predicted cancelled
Actually kept	11,754	916
Actually cancelled	1,826	2,926

The effect of duplicates. Before removing the duplicate rows, the Random Forest scored a ROC-AUC of 0.956 and caught 80% of cancellations. The drop to 0.898 suggests the earlier scores were inflated, because identical bookings appeared in both the training and test sets. The figures above are the ones to trust.

Findings
The city hotel had a higher cancellation rate (30%) than the resort hotel (23%).
Group bookings were rarely cancelled (10%), compared with 30% for transient bookings.
95% of bookings with a non-refundable deposit were cancelled, although there were only about 1,000 of them. This runs against what you would expect, so I treat it as a quirk of this dataset rather than a reliable business insight.
The Random Forest relied most on lead time (importance 0.14) and the guest's country (0.12), followed by price (0.09), arrival date (day of month, month and week number, each around 0.06-0.07), special requests (0.06) and market segment (0.05).
Feature importance shows which variables the model used, not what causes cancellations, and it does not show whether a variable raises or lowers the risk.
Limitations
The results come from a single train/test split with default model settings, so they may be slightly optimistic. I did not use cross-validation or tune the hyperparameters.
Removing exact duplicates is a judgement call. Some may be genuine, separate bookings that happen to look identical, but keeping them would risk leaking test data into training.
The bookings span several years but were split at random. A real system would be trained on past bookings and used to predict future ones.
Some columns, such as assigned_room_type, booking_changes and required_car_parking_spaces, may only be settled after a booking is made. They appear among the more important features, and I have not tested the effect of removing them.
Predictions use the default 0.5 probability cut-off. A hotel is likely to treat a missed cancellation as more costly than a false alarm, and the model currently misses 38% of cancellations, so the cut-off would need adjusting in practice.
Next steps
Use cross-validation and tune the Random Forest.
Split the data by date rather than at random.
Retrain without the possibly leaky columns and compare scores.
Try gradient boosting.
Choose the probability cut-off based on the cost of missed cancellations.
Tools

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter (VS Code).