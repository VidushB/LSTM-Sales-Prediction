import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(filepath)
dataset=dataset.drop("item_price",axis=1)
dataset['date']=pd.to_datetime(dataset['date'],format='%d.%m.%Y')
dataset=dataset.groupby(['date_block_num','shop_id','item_id'],as_index=False)["item_cnt_day"].sum() #getting sums of sales from days
dataset=dataset.pivot_table(values="item_cnt_day",index=['shop_id','item_id'],columns=["date_block_num"]) #pivoting to group based on which item at which shop
dataset.reset_index(inplace = True) #getting rid of the double layer thing
dataset=dataset.fillna(0)
dataset.drop(["shop_id","item_id"],axis=1,inplace=True)
dataset.head()
dataset.describe()

#Training data is everything but last column
X = np.expand_dims(dataset.values[:,:-1],axis = 2)
# the last column is our label
y = dataset.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
