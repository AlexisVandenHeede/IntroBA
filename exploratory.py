import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def count(df):
    counts = df.groupby(['label', 'hour', 'Date']).size().reset_index(name='count')
    print(counts)
    df = df.merge(counts, on=['label', 'hour', 'Date'], how='left')
    return df


#open weather data 2018, read textfile and convert to dataframe
text = open('data/2018_weather_data.txt', 'r')
weather_data = text.read()
weather_data = weather_data.split('\n')
weather_data = [i.split('\t') for i in weather_data[1:]]
weather_data = pd.DataFrame(weather_data)
weather_data.columns = ['Date', 'Max_temp', 'Min_temp', 'Avg_temp', 'Departure_temp', 'HDD', 'CDD', 'Precipitation', 'Snowfall', 'Snow_depth']
weather_data = weather_data.dropna()
# day month
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data['day'] = weather_data['Date'].dt.day
weather_data['month'] = weather_data['Date'].dt.month
weather_data['weekend'] = np.where(weather_data['Date'].dt.dayofweek < 5, 0, 1)
weather_data['day_of_week'] = weather_data['Date'].dt.dayofweek
weather_data['year'] = weather_data['Date'].dt.year
weather_data.drop(['Date'], axis=1, inplace=True)

print(weather_data.head())

df = pd.read_csv('data/Trips_2018.csv')
df.rename(columns={'Unnamed: 0': 'trip_id'}, inplace=True)
df.set_index('trip_id', inplace=True)
df = df.dropna()
df['starttime'] = pd.to_datetime(df['starttime'], format="%Y-%m-%d %H:%M:%S.%f")
df['stoptime'] = pd.to_datetime(df['stoptime'], format="%Y-%m-%d %H:%M:%S.%f")
df = df[~np.isnan(df['start_station_id'])]
df = df[~np.isnan(df['end_station_id'])]
# get rid of Canada outlier
df = df[df['start_station_longitude'] < -73.6]
df = df[df['end_station_longitude'] < -73.6]
df = pd.get_dummies(df, columns=['usertype'], dtype=int, drop_first=True)

print(df.columns)
print(df.head())

kmeans = KMeans(n_clusters=20, random_state=0, n_init='auto').fit(df[['end_station_latitude', 'end_station_longitude']])

df_arrival = df.drop(['start_station_latitude', 'start_station_longitude', 'start_station_id'], axis=1)
df_departure = df.drop(['end_station_latitude', 'end_station_longitude', 'end_station_id'], axis=1)
df_arrival['label'] = kmeans.predict(df[['end_station_latitude', 'end_station_longitude']])
df_departure['label'] = kmeans.predict(df[['start_station_latitude', 'start_station_longitude']])


print(df_arrival.columns)
print(df_departure.columns)
