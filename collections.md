## collections


Misc
- Goal: predict the log-error between Zillow's Zestimate and the actual sale price
  - logerror=log(Zestimate)-log(SalePrice)
  - evalutation metric: mean square error
- First round: predict 6 time points for all properties: October 2016 (201610), November 2016 (201611), December 2016 (201612), October 2017 (201710), November 2017 (201711), and December 2017 (201712). 
- Final round: predict the logerror for the months in Fall 2017.  

### dataset
* train_2016.csv: transactions file with 90,811 rows, three columns (parcelid, logerror,	transactiondate)
* train_2016_v2.csv: outliers were removed, new shape (90275, 3)
* The train data has all the transactions before October 15, 2016, plus some of the transactions after October 15, 2016.
* The test data in the public leaderboard has the rest of the transactions between October 15 and December 31, 2016.
* properties_2016.csv: list of all propperties in 2016 with their home features, 2,985,217 rows and 58 columns
* sample_submission.csv:  list of all properties with 2,985,217 and  7 columns (parcelid and 6 months prediction)


