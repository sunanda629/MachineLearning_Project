import pandas as pd
import numpy as np

df=pd.read_csv('all/train.csv')

##### For when we want to combine datasets like in the kaggle workflow
# train_df = pd.read_csv('all/train.csv')
# test_df = pd.read_csv('all/test.csv')
#
# # Save the 'Id' column
# train_ID = train_df['Id']
# test_ID = test_df['Id']
#
# # Now drop the 'Id' colum since we can not use it as a feature to train our model.
# train_df.drop("Id", axis = 1, inplace = True)
# test_df.drop("Id", axis = 1, inplace = True)
#
# y_train = train_df['SalePrice']
# X_train = train_df.drop('SalePrice', axis=1)
# X_test = test_df.copy()
#
# del train_df, test_df
#
# df = pd.concat([X_train, X_test], ignore_index=True)
# df.shape

one_hot_df = pd.get_dummies(df, drop_first=True, dummy_na=True) # new column will be DummyNA :/
n,p=one_hot_df.shape
n/p #since this is <20:1 ratio, we should one hot encode this dataset
del one_hot_df,n,p

# let's give the ordinal columns numbers
ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual',
           'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}

for col in ord_cols:
    df[col] = df[col].map(lambda x: ord_dic.get(x, 0))
df.shape
n,p=df.shape
n/p
df.loc[df.Heating.isnull()]
# %% Imputing LotFrontage

dfF=df[[ 'LotFrontage', 'LotArea', 'LotConfig', 'LotShape']]
dfF=pd.get_dummies(dfF,drop_first=True,dummy_na=True)
train=dfF.loc[dfF.LotFrontage.isnull()==False]
test=dfF.loc[dfF.LotFrontage.isnull()==True]
test=test.drop('LotFrontage',axis=1)

train2.sample(10)
n,p=df.shape
n/p # good shape for dummification
# %%
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
X=train2.drop('LotFrontage',axis=1)
X=X.astype(float)
Y=train2.LotFrontage.astype(float)
print(X.shape,Y.shape)
ols.fit(X, Y)
y_predict = ols.predict(test)

print("beta_1, beta_2: " + str(np.round(ols.coef_, 3)))
print("beta_0: " + str(np.round(ols.intercept_, 3)))
# not sure that this RSS is correct
print("RSS: %.2f" % np.sum((y_predict - Y.mean()) ** 2))
print("R^2: %.5f" % ols.score(X, Y))

# %% Separating out my Separating

#dataframeSophie

dfS=df[['SalePrice','Heating','HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', \
'2ndFlrSF', \
'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', \
'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', \
'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu']]

# %% Missingness
dfS.dtypes
dfS=dfS.dropna(subset=['Electrical'])
dfS.BsmtFullBath=dfS.BsmtFullBath.fillna(0.0)
dfS.BsmtHalfBath=dfS.BsmtHalfBath.fillna(0.0)
dfS.Functional=dfS.Functional.fillna('Typ')

miss=dfS.loc[:,dfS.isnull().sum()!=0]
dfS.Functional.unique()
dfS.loc[dfS.Functional.isnull()]
miss.columns

# %% Plotting variables
import matplotlib.pyplot as plt


def sctplot(x,i):
    print(str(num_colnames[i]))
    plt.scatter(x,dfS['SalePrice'])
    plt.title('SalePrice vs ' + str(num_colnames[i]))
    return plt.show()

num_colnames=dfS.describe().columns[1:-1]
[sctplot(dfS[num_colnames[i]],i) for i in range(0,len(num_colnames))]
