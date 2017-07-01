## collections


Misc
- Goal: predict the log-error between Zillow's Zestimate and the actual sale price
  - logerror=log(Zestimate)-log(SalePrice)
- predict the logerror for the months in Fall 2017.  

### dataset
* train_2016.csv: transactions file with 90811 rows, three columns (parcelid, logerror,	transactiondate)
* The train data has all the transactions before October 15, 2016, plus some of the transactions after October 15, 2016.
* The test data in the public leaderboard has the rest of the transactions between October 15 and December 31, 2016.


