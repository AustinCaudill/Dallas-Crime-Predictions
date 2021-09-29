""" 
Dallas Crime Predictions
Sept 28th 2021
Austin Caudill

"""

# Begin Timer for script
import time
start_time = time.time()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import geopandas as gpd
import numpy as np
import folium
from folium import plugins
import lightgbm as lgbm
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from pdpbox import pdp, get_dataset, info_plots
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Imports Loaded Successfully.")
# ===============================

data = pd.read_csv("Police_Incidents.csv", parse_dates=['Time1 of Occurrence', 'Date1 of Occurrence', 'Year1 of Occurrence'])
# Columns 3, 41, 52, 57, 60, 74 have mixed data types. Examine.
# print(data.iloc[:, [3, 3]])



# Rename columns for ease of use.
data.rename(columns={'Date1 of Occurrence': 'Date'}, inplace=True)
data.rename(columns={'Year1 of Occurrence': 'Year'}, inplace=True)

# Good records weren't kept until mid-2014, so only want to keep records for subsequent years.
data = data[data.Year >= "2015"]

# Check to ensure records were removed.
print('First date: ', str(data.Date.describe(datetime_is_numeric=True)['min']))
print('Last date: ', str(data.Date.describe(datetime_is_numeric=True)['max']))

# Check for duplicate records.
print('Number of duplicate records: %s'  % (data['Incident Number w/year'].duplicated().sum()))  # Someday should convert to ftsring format.
# sorting by incident-year
data.sort_values("Incident Number w/year", inplace = True)
# dropping ALL duplicate values
data.drop_duplicates(subset ="Incident Number w/year",
                     keep = False, inplace = True)
print('Number of duplicate records: %s'  % (data['Incident Number w/year'].duplicated().sum()))  # Check again
# Duplicates have been removed.

# Need to extract Lat and Long from column: Location1
data['Lat_and_Long'] = data['Location1'].str.extract(r'\(([^)]+)')
# Now need to split into seperate columns and store in dataframe
temp = data['Lat_and_Long'].str.strip('()').str.split(', ', expand=True).rename(columns={0:'Latitude', 1:'Longitude'}) 
data = pd.concat([data, temp], axis=1)
# To free up memory:
del temp


# Analyzing number of incidents per day
col = sns.color_palette()

plt.figure(figsize=(10, 6))
dates = data.groupby('Date').count().iloc[:, 0]
sns.kdeplot(data=dates, shade=True)
plt.axvline(x=dates.median(), ymax=0.95, linestyle='--', color=col[2])
plt.annotate(
    'Median: ' + str(dates.median()),
    xy=(dates.median(), 0.004),
    xytext=(200, 0.005),
    arrowprops=dict(arrowstyle='->', color=col[1], shrinkB=10))
plt.title('Distribution of Number of Incidents per Day', fontdict={'fontsize': 16})
plt.xlabel('Incidents')
plt.ylabel('Density')
plt.show()

# Analyzing number of incidents per year
plt.figure(figsize=(10, 6))
years = data.groupby('Year').count().iloc[:, 0]
sns.kdeplot(data=years, shade=True)
plt.axvline(x=years.median(), ymax=2, linestyle='--', color=col[2])
plt.annotate(
    'Median: ' + str(years.median()),
    xy=(years.median(), 0.004),
    xytext=(200, 0.005),
    arrowprops=dict(arrowstyle='->', color=col[1], shrinkB=10))
plt.title('Distribution of Number of Incidents per Year', fontdict={'fontsize': 16})
plt.xlabel('Incidents')
plt.ylabel('Density')
plt.show()


byday = data.groupby('Day1 of the Week').count().iloc[:, 0]
byday = byday.reindex(['Sun','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        byday.index, (byday.values / byday.values.sum()) * 100,
        orient='v',
        palette=cm.ScalarMappable(cmap='Reds').to_rgba(byday.values))
plt.title('Incidents per Weekday', fontdict={'fontsize': 16})
plt.xlabel('Weekday')
plt.ylabel('Incidents (%)')
plt.show()


byincident = data.groupby('NIBRS Crime Category').count().iloc[:, 0].sort_values(ascending=False).head(15)
byincident = byincident.reindex(np.append(np.delete(byincident.index, 0), 'MISCELLANEOUS'))
plt.figure(figsize=(10, 10))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        (byincident.values / byincident.values.sum()) * 100,
        byincident.index,
        orient='h',
        palette="Reds_r")
plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})
plt.xlabel('Incidents (%)')
plt.show()

# Mapping Lat and Long points - Not useful due to amnount of points plotted.
long_min = -97.0690
long_max = -96.4593
lat_min = 33.0333
lat_max = 32.6006
BBox = (long_min, long_max, lat_min, lat_max)
Dallas_Map = plt.imread("Map_of_Dallas.png")
fig, ax = plt.subplots(figsize = (8,7))
data['Latitude'] = pd.to_numeric(data['Latitude'])
data['Longitude'] = pd.to_numeric(data['Longitude'])
ax.scatter(data['Longitude'], data['Latitude'], zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting Spatial Data on Dallas Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(Dallas_Map, zorder=0, extent = BBox, aspect= 'equal')

# Heatmap by Lat/Long points
print('All Crime')
heatmap_data = data[['Latitude', 'Longitude']].dropna()
heatmap = folium.Map([32.7767, -96.7970], zoom_start=11)
heatmap.add_child(plugins.HeatMap(heatmap_data, radius=15))
display(heatmap)
del heatmap_data # To conserve resources

##########################################
# Geographic Density of Different Crimes
crimes = data['NIBRS Crime Category'].unique().tolist()
for i in crimes:
    print(i)
    heatmap_base = folium.Map([32.7767, -96.7970], zoom_start=11)
    # Ensure you're handing it floats
    data['Latitude'] = data['Latitude'].astype(float)
    data['Longitude'] = data['Longitude'].astype(float)
    # Filter the DF for rows, then columns, then remove NaNs
    heat_df = data[['Latitude', 'Longitude', 'NIBRS Crime Category']]
    heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])
    heat_df = heat_df.loc[heat_df['NIBRS Crime Category'] == i]
    # List comprehension to make out list of lists
    heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]
    # Plot it on the map
    heatmap_base.add_child(plugins.HeatMap(heat_data, radius=15))
    # Display the map display(heatmap_base)
    print()

##########################################
data['Time for Hour'] = data['Time1 of Occurrence'].copy()
data['Hour'] = data['Time for Hour'].dt.hour
data['Time for Min'] = data['Time1 of Occurrence'].copy()
data['Minute'] = data['Time for Min'].dt.minute
data['Time for Mon'] = data['Date'].copy()
data['Month'] = data['Time for Mon'].dt.month

data3 = data.groupby(['Hour', 'Date', 'NIBRS Crime Category'], as_index=False).count().iloc[:,:4]
data3.rename(columns={'Incident Number w/year': 'Incidents'}, inplace=True)
data3 = data3.groupby(['Hour', 'NIBRS Crime Category'], as_index=False).mean()
data3 = data3.loc[data3['NIBRS Crime Category'].isin(
   ['LARCENY/ THEFT OFFENSES', 'DESTRUCTION/ DAMAGE/ VANDALISM OF PROPERTY', 'ASSAULT OFFENSES', 'BURGLARY/ BREAKING & ENTERING', 'PUBLIC INTOXICATION'])]
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))
ax = sns.lineplot(x='Hour', y='Incidents', data=data3, hue='NIBRS Crime Category')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
plt.suptitle('Average number of incidents per hour')
plt.xticks(range(0, 24))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

### Machine Learning Time ###
# Keep only needed columns
newdata = data[['Month','Minute','Hour','Division','NIBRS Crime Category','Year of Incident','Day1 of the Year','Watch','Beat','Latitude','Longitude','Zip Code']].copy()
newdata = newdata[newdata.Watch != 'U']
newdata['Watch'].dropna(inplace=True)
newdata['Watch'] = newdata['Watch'].astype(str).astype(int)

# Split dataframe into train and test datasets.
train, test = train_test_split(newdata, test_size=0.33, random_state=42)
# Seperate the target (`y`) from the training features (`features`).

# Separate target from features
y = train['NIBRS Crime Category']
features = train.drop(['NIBRS Crime Category'], axis=1)

# List of features for later use
feature_list = list(features.columns)

# Preview features
features.head()

# label-encode categorical columns
X = features.copy()
X_test = test.copy()

# Need to Label Encode
labelencoder = LabelEncoder()
X['Division'] = labelencoder.fit_transform(features['Division'])
X_test['Division'] = labelencoder.transform(test['Division'])
X_test = X_test.drop(['NIBRS Crime Category'], axis=1)

le2 = LabelEncoder()
y= le2.fit_transform(train['NIBRS Crime Category'])
# Need key for later reference - Note: These are listed alphabetically.
print(dict(zip(le2.classes_,range(len(le2.classes_)))))


# Split validation set from the training data.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

model = LGBMClassifier() 

# Train the model
model.fit(X_train,y_train)
preds_valid = model.predict(X_valid)

# Make predictions on entire data set.
predictions = model.predict(X_test)

perm = PermutationImportance(model).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names=X_test.columns.tolist())

# Creating the model
y_pred=model.predict(X_valid)




# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_valid)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_valid, y_pred)))

model = LGBMClassifier().fit(X, y, categorical_feature=['Division'])

pdp_Pd = pdp.pdp_isolate(
    model=model,
    dataset=X,
    model_features=X.columns.tolist(),
    feature='Hour',
    n_jobs=-1)

pdp.pdp_plot(
    pdp_Pd,
    'Hour',
    ncols=3)
plt.show()



# Test it out
data_for_prediction = X_test.loc[[96623]]
print(data_for_prediction)

shap.initjs()

# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[6], shap_values[5], data_for_prediction, link='logit')


# ===============================
print("Success")
print("--- %s seconds ---" % (time.time() - start_time))