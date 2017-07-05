import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error as mae
import os

#%%

path2traindata='../../data/train_2016_v2.csv'
path2property='../../data/properties_2016.csv'
path2sample='../../data/sample_submission.csv'
path2submission='./output/submissions/'

#%%

train = pd.read_csv(path2traindata)
properties = pd.read_csv(path2property)

for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.4 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

#%%
# xgboost params
xgb_params = {
    'eta': 0.07,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# train validation split
x_train=np.array(x_train)
n_folds=10
skf = list(StratifiedKFold(y_train, n_folds))
 
# xgboost classifier
clf = xgb.XGBRegressor(max_depth=3,
                       n_estimators=150,
                       min_child_weight=8,
                       learning_rate=0.06,
                       nthread=8,
                       subsample=0.70,
                       colsample_bytree=0.80,
                       seed=4241)

# train and validation
loss=[]
y_test_pred=np.zeros(len(x_test))
for i, (train, val) in enumerate(skf):
        print "Fold", i
        x_train_i = x_train[train]
        y_train_i = y_train[train]
        x_val = x_train[val]
        y_val = y_train[val]
        clf.fit(x_train_i, y_train_i, eval_set=[(x_val, y_val)], verbose=True, eval_metric='mae', early_stopping_rounds=20)
        y_val_pred=clf.predict(x_val,ntree_limit=clf.best_iteration)
        loss.append(mae(y_val,y_val_pred))
        y_test_pred+=clf.predict(np.array(x_test),ntree_limit=clf.best_iteration)

print 'average loss: %.5f' %(np.mean(loss))
            
#%%
#pred = model.predict(dtest)
y_pred=[]

# average
pred=y_test_pred/n_folds

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

#from datetime import datetime
#output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

#%% create sumission

#sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')
import datetime
now = datetime.datetime.now()
info='xgb_regressor_nfolds'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join(path2submission, 'submission_' + suffix + '.csv')

output.to_csv(sub_file, index=False, float_format='%.4f')
print(output.head())    

