import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def count(df):
    counts = df.groupby(['label', 'hour']).size().reset_index(name='count')
    df = df.merge(counts, on=['label', 'hour'], how='left')
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2   
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # for integers
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64) 
            else:  # for floats.
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)   
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
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

weather_data.replace('T', 0, inplace=True)
weather_data.replace('M', 0, inplace=True)

# check for nan
print(weather_data.isnull().sum())

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

df_arrival = df.drop(['start_station_latitude', 'start_station_longitude', 'start_station_id', 'starttime'], axis=1)
df_departure = df.drop(['end_station_latitude', 'end_station_longitude', 'end_station_id', 'stoptime'], axis=1)

# make lat and long called that and time
df_arrival.rename(columns={'end_station_latitude': 'latitude', 'end_station_longitude': 'longitude'}, inplace=True)
df_departure.rename(columns={'start_station_latitude': 'latitude', 'start_station_longitude': 'longitude'}, inplace=True)
df_arrival.rename(columns={'stoptime': 'time'}, inplace=True)
df_departure.rename(columns={'starttime': 'time'}, inplace=True)

print(df_arrival.columns)
print(df_departure.columns)

# add hour, day, month, weekend, day of week
df_arrival['hour'] = df_arrival['time'].dt.hour
df_departure['hour'] = df_departure['time'].dt.hour
df_arrival['day'] = df_arrival['time'].dt.day
df_departure['day'] = df_departure['time'].dt.day
df_arrival['month'] = df_arrival['time'].dt.month
df_departure['month'] = df_departure['time'].dt.month
df_arrival['weekend'] = np.where(df_arrival['time'].dt.dayofweek < 5, 0, 1)
df_departure['weekend'] = np.where(df_departure['time'].dt.dayofweek < 5, 0, 1)
df_arrival['day_of_week'] = df_arrival['time'].dt.dayofweek
df_departure['day_of_week'] = df_departure['time'].dt.dayofweek

kmeans = KMeans(n_clusters=20, random_state=0, n_init='auto').fit(df_arrival[['latitude', 'longitude']])

# add labels using k means predict
df_arrival['label'] = kmeans.predict(df_arrival[['latitude', 'longitude']])
df_departure['label'] = kmeans.predict(df_departure[['latitude', 'longitude']])

# count
df_arrival = count(df_arrival)
df_departure = count(df_departure)
print('added count')


# drop duplicates
df_arrival = df_arrival.drop_duplicates()
df_departure = df_departure.drop_duplicates()



# # merge weather data with arrival and departure
# df_arrival = df_arrival.merge(weather_data, on=['day', 'month', 'day_of_week', 'weekend'], how='left')
# df_departure = df_departure.merge(weather_data, on=['day', 'month', 'day_of_week', 'weekend'], how='left')

# # check for nan
# print(df_arrival.isnull().sum())
# print(df_departure.isnull().sum())
# # drop nan
# df_arrival.dropna(inplace=True)
# df_departure.dropna(inplace=True)

# print(df_arrival.columns)
# print(df_departure.columns)

# # print type per column
# print(df_arrival.dtypes)
# print(df_departure.dtypes)

# # drop time
# df_arrival.drop(['time'], axis=1, inplace=True)
# df_departure.drop(['time'], axis=1, inplace=True)

# # convert to float16 if column data type  is object
# for col in df_arrival.columns:
#     if df_arrival[col].dtype == 'object':
#         df_arrival[col] = df_arrival[col].astype('float16')
# for col in df_departure.columns:
#     if df_departure[col].dtype == 'object':
#         df_departure[col] = df_departure[col].astype('float16')

# print(df_arrival.dtypes)
# print(df_departure.dtypes)

# print('reducing memory')
# df_arrival = reduce_mem_usage(df_arrival)
# df_departure = reduce_mem_usage(df_departure)

# # standardise data
# print('standardising')
# scaler = StandardScaler()
# df_arrival = scaler.fit_transform(df_arrival)
# df_departure = scaler.fit_transform(df_departure)


# # split into train and test and eval
# split = 0.2
# df_arrival_train, df_arrival_test = train_test_split(df_arrival, test_size=split)
# df_arrival_train, df_arrival_eval = train_test_split(df_arrival_train, test_size=split)
# df_departure_train, df_departure_test = train_test_split(df_departure, test_size=split)
# df_departure_train, df_departure_eval = train_test_split(df_departure_train, test_size=split)

# print('saving now')

# # save to csv
# np.savetxt('data/arrival_train_count.csv', df_arrival_train, delimiter=',')
# np.savetxt('data/arrival_test_count.csv', df_arrival_test, delimiter=',')
# np.savetxt('data/arrival_eval_count.csv', df_arrival_eval, delimiter=',')
# np.savetxt('data/departure_train_count.csv', df_departure_train, delimiter=',')
# np.savetxt('data/departure_test_count.csv', df_departure_test, delimiter=',')
# np.savetxt('data/departure_eval_count.csv', df_departure_eval, delimiter=',')

# # stream line by finding count, then do unique then merge
