""" 
Dallas Crime Predictions
Sept 27th 2021
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
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier

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
byday = byday.reindex([
    'Sun','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
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
    # Display the map
    display(heatmap_base)
    print()

##########################################
data['Hour'] = data['Time1 of Occurrence'].dt.hour
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
# Drop unneeded columns
# data.drop(columns=['Incident Number w/year', 'Service Number ID', 'Call (911) Problem', 'Incident Address', 'Apartment Number', 'Target Area Action Grids', 'Community', 'Date of Report', 'Date incident created', 'Offense Entered Month', 'Offense Entered Day of the Week', 'Offense Entered Time', 'CFS Number', 'Call Received Date Time', 'Call Date Time', 'Call Cleared Date Time', 'Call Dispatch Date Time', 'Special Report (Pre-RMS)', 'Person Involvement Type', 'Victim Type', 'Victim Name', 'Victim Race', 'Victim Ethnicity', 'Victim Gender', 'Victim Home Address', 'Victim Apartment', 'Victim Zip Code', 'Victim City', 'Victim State', 'Victim Business Name', 'Victim Business Address', 'Victim Business Phone', 'Responding Officer #1  Badge No', 'Responding Officer #1  Name', 'Responding Officer #2 Badge No', 'Responding Officer #2  Name', 'Reporting Officer Badge No', 'Assisting Officer Badge No', 'Reviewing Officer Badge No', 'Element Number Assigned', 'Investigating Unit 1', 'Investigating Unit 2', 'Offense Status', 'UCR Disposition', 'Victim Injury Description', 'Victim Condition', 'Hate Crime Description', 'Drug Related Istevencident', 'RMS Code', 'Penal Code', 'UCR Offense Name', 'UCR Offense Description', 'Update Date', 'Lat_and_Long'], axis=1, inplace=True)
newdata = data[['Hour','Division','NIBRS Crime Category','Year of Incident','Beat']].copy()
#data.drop(columns=['Watch', 'Type of Incident', 'Type  Location', 'Type of Property', 'Council District', 'Date', 'Year', 'Month1 of Occurence', 'Day1 of the Week', 'Time1 of Occurrence', 'Month2 of Occurence', 'Day2 of the Week', 'Time2 of Occurrence', 'Modus Operandi (MO)', 'Family Offense', 'Hate Crime', 'Weapon Used', 'Gang Related Offense', 'Offense Type', 'NIBRS Crime', 'NIBRS Crime Against', 'NIBRS Code', 'NIBRS Group', 'NIBRS Type', 'City', 'State', 'Location1'], axis=1, inplace=True)
# Split dataframe into train and test datasets.
train, test = train_test_split(newdata, test_size=0.33, random_state=42)
# feature selection
from sklearn.preprocessing import LabelEncoder
# Encoding the Categorical Variables
le1 = LabelEncoder()
train['Division'] = le1.fit_transform(train['Division'])
test['Division'] = le1.transform(test['Division'])
le2 = LabelEncoder()
y = le2.fit_transform(train.pop('NIBRS Crime Category'))

train_X, val_X, train_y, val_y = train_test_split(train, y)

model =LGBMClassifier(objective='multiclass', num_class=39).fit(train_X, train_y)

perm = PermutationImportance(model).fit(val_X, val_y)
print(eli5.show_weights(perm, feature_names=val_X.columns.tolist()))

# Creating the model
y_pred=model.predict(val_X)
# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, val_y)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(val_y, y_pred)))










# ===============================
print("Success")
print("--- %s seconds ---" % (time.time() - start_time))