
## Data loading
# %%
%load_ext autoreload
%autoreload 2

from house import *
del house
house = House('data/train.csv','data/test.csv')
# %%



##EDA:

#Response varriable:
house.train().SalePrice.describe()
house.log_transform(house.train().SalePrice)
# house.corr_matrix(house.train(), 'SalePrice')

#Missing values:
#house.missing_stats()
house.clean()

columns_to_convert = [  ('MSSubClass', 'object'), ('LotArea', 'float64' ), ('OverallQual', 'object'),
                        ('OverallCond', 'object'), ('1stFlrSF', 'float64'), ('2ndFlrSF', 'float64'),
                        ('3SsnPorch', 'float64'), ('EnclosedPorch', 'float64'), ('GarageCars', 'int64'),
                        ('WoodDeckSF', 'float64'), ('ScreenPorch', 'float64'), ('OpenPorchSF', 'float64'),
                        ('MiscVal', 'float64'), ('LowQualFinSF', 'float64'), ('GrLivArea', 'float64'), ]

house.convert_types(columns_to_convert)

house.distribution_charts()

# now check skewness and inspect distribution charts again
house.sg_skewness()

for var in house.skewed_features:
    house.log_transform(house.train()[var])
house.log_transform(house.train()['1stFlrSF'])

house.distribution_charts()

house.sg_ordinals()
house.label_encode_engineer()
# house.label_df.sample(10)


##Feature Engeneneering:
house.engineer_features()


house.sk_random_forest(500)

house.sg_ord_random_forest(500)
