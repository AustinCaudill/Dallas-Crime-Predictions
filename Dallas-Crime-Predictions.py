""" 
Dallas Crime Predictions
Sept 22nd 2021
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

from sklearn.model_selection import train_test_split

print("Imports Loaded Successfully.")
# ===============================

data = pd.read_csv("Police_Incidents.csv", parse_dates=['Date1 of Occurrence', 'Year1 of Occurrence'])
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

# Need to remove missing location data at some point.
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
heatmap_data = data[['Latitude', 'Longitude']].dropna()
heatmap = folium.Map([32.7767, -96.7970], zoom_start=11)
heatmap.add_child(plugins.HeatMap(heatmap_data, radius=15))
display(heatmap)
del heatmap_data

# Heatmap by Zip Code
# count number of incidences grouped by zipcode
# Set zipcode type to string (folium)
# data['Zip Code'] = data['Zip Code'].astype('str')
# get the mean value across all data points
zipcode_data = data.groupby('Zip Code').aggregate(np.mean)
zipcode_data.reset_index(inplace = True)
data['count'] = 1
temp = data.groupby('Zip Code').sum()
temp.reset_index(inplace = True)
temp = temp[['Zip Code', 'count']]
zipcode_data = pd.merge(zipcode_data, temp, on='Zip Code')
# read updated geo data
import urllib.request
print('Beginning file download with urllib2...')
url = 'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/tx_texas_zip_codes_geo.min.json'
urllib.request.urlretrieve(url, 'zipcode.json')
# load GeoJSON
import simplejson as json
# Get geo data file path
with open('zipcode.json') as f:
    txArea = json.load(f)
geozips = []
# zipcode_data['Zip Code'] = zipcode_data['Zip Code'].astype(np.int)
# zipcode_data = zipcode_data.iloc[1: , :]
for i in range(len(txArea['features'])):
    if txArea['features'][i]['properties']['ZCTA5CE10'] in list(zipcode_data['Zip Code']):
        geozips.append(txArea['features'][i])
# creating new JSON object
new_json = dict.fromkeys(['type','features'])
new_json['type'] = 'FeatureCollection'
new_json['features'] = geozips
# save uodated JSON object
open("cleaned_geodata.json", "w").write(json.dumps(new_json, sort_keys=True, indent=4, separators=(',', ': ')))
laMap = folium.Map(location=[32.7767, -96.7970], zoom_start=11)
#add the shape of LA County to the map
folium.GeoJson(txArea).add_to(laMap)
laMap.choropleth(geo_path=geozips, geo_data=zipcode_data, columns=['Zip Code', 'count'], \
    key_on='features.properties.ZCTA5CE10', fill_color='YlGn', fill_opacity=1)
display(laMap)

# Split dataframe into train and test datasets.
# train, test = train_test_split(data, test_size=0.33, random_state=42)



# ===============================
print("Success")
print("--- %s seconds ---" % (time.time() - start_time))