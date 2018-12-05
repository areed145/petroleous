import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

brent = pd.read_excel('kaggle/RBRTEd.xls', sheet_name='Data 1')
brent.columns=brent.iloc[1]
brent=brent.ix[2:]
brent['Date']=brent['Date'].astype('datetime64[ns]')
brent.columns=['date','oil_price']
brent.head()

brent.info()

# Loop to read all share price files and transform data
shares=['RDSB.L','BP.L','CNE.L','PMO.L','STL.OL','FP.PA','REP.MC','ENGI.PA','SLB.PA']

all_data=pd.DataFrame() # here we will store all the data from all shares and oil price in a master dataframe
for index in range(len(shares)):
    stock=pd.DataFrame()

	# 1.- Read files
    stock=pd.read_csv('kaggle/'+shares[index]+'.csv')     
	# 2.- Transform data
    stock=stock[['Date','Close']]       
    stock['Date']=stock['Date'].astype('datetime64[ns]')
    stock.columns=['date','share_price']
    test=pd.DataFrame(brent) # VLOOKUP equivalent in Python to merge 2 sets of data
    output=stock.merge(test,on='date',how='left')
    stock['oil_price']=output['oil_price']
    stock['share_price']=pd.to_numeric(stock['share_price'], errors='coerce').dropna(0)
    stock['oil_price']=pd.to_numeric(stock['oil_price'], errors='coerce').dropna(0)
    stock['year']=pd.to_datetime(stock['date']).dt.year # Create a column with the year to filter later
    stock['name']=shares[index]
    stock = stock.dropna() # get rid of all the NAN rows.
	# 3.- Feature Engineering. Create new column with scaled share price from 0 to 1.
	#     This will help us comparing companies later on.
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    stock['share_price_scaled']=scaler.fit_transform(stock['share_price'].to_frame())
	# 4.- Append data to a master dataframe
    all_data=all_data.append(stock) #append data to one matrix
    
all_data.head()

brent[['date','oil_price']].set_index('date').plot(color='green', linewidth=1.0)

#==============================================================================
# Pairplot using master data table (all_data) with a filter on BP share price
#==============================================================================
palette=sns.cubehelix_palette(18, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(all_data[all_data['name']=='BP.L'].drop(['share_price_scaled'],axis=1),
             hue='year',palette=palette,size=4,markers='o',
             plot_kws=dict(s=50, edgecolor='b', linewidth=0))

#==============================================================================
# Pairplot on less data 2013 to 2017 using Royal Dutch Shell (LON) stock price
#==============================================================================

# Just for the last 5 years
all_data13=all_data[all_data['year']>2012]
palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(all_data13[all_data13['name']=='RDSB.L'].drop(['share_price_scaled'],axis=1),
             hue='year',palette=palette,size=4,markers='o',
             plot_kws=dict(s=50, edgecolor='b', linewidth=0))

#==============================================================================
# Violin Plot Oil price on last 5 years
#==============================================================================

sns.set_style('whitegrid')
palette=sns.cubehelix_palette(5, start=2.8, rot=0, dark=0.2, light=0.8, reverse=False)

sns.violinplot(x='year', y='oil_price', data=all_data13[all_data13['name']=='RDSB.L'],
               inner='quart', palette=palette, trim=True)

#==============================================================================
# Violin Plot Oil price on last 5 years
#==============================================================================

sns.factorplot(x='year', y='share_price_scaled', col='name', col_wrap=3,kind='violin',
               split=True, data=all_data13,inner='quart', palette=palette, trim=True,size=4,aspect=1.2)
sns.despine(left=True)

#==============================================================================
# joint plot using 5 years for Premier Oil
#==============================================================================

sns.jointplot('oil_price', 'share_price',data=all_data13[all_data13['name']=='PMO.L'],kind='kde',
              hue='year',size=6,ratio=2,color='red').plot_joint(sns.kdeplot, zorder=0, n_levels=20)

#==============================================================================
# joint plot using 5 years for Statoil
#==============================================================================

sns.jointplot('oil_price', 'share_price',data=all_data13[all_data13['name']=='STL.OL'],kind='kde',
              hue='year',size=6,ratio=2,color='blue').plot_joint(sns.kdeplot, zorder=0, n_levels=20)

#==============================================================================
# lmplot using using 5 years for all companies
#==============================================================================

sns.lmplot(x='oil_price', y='share_price_scaled', col='name',ci=None, col_wrap=3, 
           data=all_data13, order=1,line_kws={'color': 'blue'},scatter_kws={'color': 'grey'}).set(ylim=(0, 1))

palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.lmplot(x='oil_price', y='share_price_scaled',hue='year', col='name',ci=None, 
           col_wrap=3, data=all_data13, order=1,palette=palette,size=4).set(ylim=(0, 1))

#==============================================================================
# Unsupervised Learning - Cluster analysis on Shell data
#==============================================================================
from sklearn.cluster import KMeans

shell=pd.DataFrame()
shell=all_data13[all_data13['name']=='RDSB.L']
# We need to scale also oil price, so clustering is not influenced by the relative size of one axis.
shell['oil_price_scaled']=scaler.fit_transform(shell['oil_price'].to_frame())
shell['cluster'] = KMeans(n_clusters=6, random_state=1).fit_predict(shell[['share_price_scaled','oil_price_scaled']])

# The 954 most common RGB monitor colors https://xkcd.com/color/rgb/
colors = ['baby blue', 'amber', 'scarlet', 'grey','milk chocolate', 'windows blue']
palette=sns.xkcd_palette(colors)

sns.lmplot(x='oil_price', y='share_price_scaled',ci=None,palette=palette, hue='cluster',fit_reg=0 ,data=shell)

#==============================================================================
# Supervised learning linear regression
#==============================================================================

from sklearn import linear_model

# 1.- Data preparation
shell15=pd.DataFrame()
shell15=all_data13[(all_data13['name']=='RDSB.L') & (all_data13['year']>2015 )] # Extract data from years 2016/17
shell15=shell15[['share_price','oil_price']].reset_index()

# Just using 1 variable for linear regression. To try with more variables use randomforest
# Split the data into training/testing sets
train = shell15[:-100]
test = shell15[-100:]

x_train=train['oil_price'].to_frame() #converts the pandas Series to numpy.ndarray
y_train=train['share_price'].to_frame()
x_test=test['oil_price'].to_frame() #converts the pandas Series to numpy.ndarray
y_test=test['share_price'].to_frame()

# 2.- Create linear regression object
regr = linear_model.LinearRegression()

# 3.- Train the model using the training sets
regr.fit(x_train,y_train)

# The coefficients
print('Coefficients: ',  float(regr.coef_))
# The mean squared error
print('Mean squared error: %.2f'
      % np.mean((regr.predict(x_train) - y_train) ** 2))

# Plot outputs using matplotlib
plt_train=plt.scatter(x_train, y_train,  color='grey')
plt_test=plt.scatter(x_test, y_test,  color='green')
plt.plot(x_train, regr.predict(x_train), color='black', linewidth=3)
plt.plot(x_test,regr.predict(x_test),  color='black', linewidth=3)
plt.xlabel('oil_price')
plt.ylabel('share_price')
plt.legend((plt_train, plt_test),('train data', 'test data'))
plt.show()

from sklearn.ensemble import RandomForestRegressor
# 1.- Data Preparation
shell15=pd.DataFrame()
shell15=all_data13[(all_data13['name']=='RDSB.L') & (all_data13['year']>2015 )]
shell15=shell15[['share_price','oil_price']].reset_index()

# Load share price of other variables
shell15['PMO.L']=all_data13[(all_data13['name']=='PMO.L')][-373:].reset_index()['share_price']
shell15['CNE.L']=all_data13[(all_data13['name']=='CNE.L')][-373:].reset_index()['share_price']
shell15['FP.PA']=all_data13[(all_data13['name']=='FP.PA')][-373:].reset_index()['share_price']
shell15['ENGI.PA']=all_data13[(all_data13['name']=='ENGI.PA')][-373:].reset_index()['share_price']

train = shell15[:-100]
test = shell15[-100:]

x_train=train[['oil_price','PMO.L','CNE.L','FP.PA','ENGI.PA']]
y_train=train['share_price']

x_test=test[['oil_price','PMO.L','CNE.L','FP.PA','ENGI.PA']] 
y_test=test['share_price'].to_frame()


# 2.- Create Randomforest object usinig a max depth=5
regressor = RandomForestRegressor(n_estimators=200, max_depth=5 )

# 3.- Train data
clf=regressor.fit(x_train, y_train)

# 4.- Predict!
y_pred=regressor.predict(x_test)
y_pred=pd.DataFrame(y_pred)

# We are going to have a look at how fitted data looks like:

plt_train=plt.scatter(x_train['oil_price'],y_train,   color='grey')
plt_pred=plt.scatter(shell15['oil_price'], regressor.predict(shell15[['oil_price','PMO.L','CNE.L','FP.PA','ENGI.PA']]),  color='black')

plt.xlabel('oil_price')
plt.ylabel('share_price')
plt.legend((plt_train,plt_pred),('train data','prediction'))
plt.show()

plt_train=plt.scatter(x_train['oil_price'],y_train,   color='grey')
plt_test=plt.scatter(x_test['oil_price'],y_test,   color='green')
plt_pred=plt.scatter(x_test['oil_price'], y_pred,  color='black')

plt.xlabel('oil_price')
plt.ylabel('share_price')
plt.legend((plt_train, plt_test,plt_pred),('train data', 'test data','prediction'))
plt.show()

# The mean squared error
print('Mean squared error: %.2f'
      % np.mean((regressor.predict(x_train) - y_train) ** 2))

importances=regressor.feature_importances_

indices=list(x_train)
print('Feature ranking:')

for f in range(x_train.shape[1]):
    print('Feature %s (%f)' % (indices[f], importances[f]))

f, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.barplot(indices, importances, palette='BrBG', ax=ax1)
ax1.set_ylabel('Importance')