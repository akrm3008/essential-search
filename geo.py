import googlemaps
import pandas as pd
import math

key = pd.read_csv('key.csv')
gmaps = googlemaps.Client(key= key.columns[0])


def getGeoLocation(df, city, gmaps):
    df_city= df[df['CITY'] == city]
    Add = df_city['ADDRESS'] + ',' + " " +  df_city['CITY'] 
    geocode_result = Add.apply(lambda x: gmaps.geocode(x))
    df_city['LAT'] = geocode_result.apply(lambda x : x[0]["geometry"]["location"]["lat"])
    df_city['LONG'] = geocode_result.apply(lambda x : x[0]["geometry"]["location"]["lng"])
    df_city['Location']= geocode_result.apply(lambda x : [x[0]["geometry"]["location"]["lat"],
                                              x[0]["geometry"]["location"]["lng"]])
    return df_city


def getGasStationWithinRadius(df_city, lat, lon, r, gmaps):
    radius = 3958.8 # Earth radius
    lat1 = df_city['LAT']*math.pi/180
    lat2 = lat*math.pi/180
    lon1 = df_city['LONG']*math.pi/180
    lon2 = lon*math.pi/180
    deltaLat = lat1 - lat2
    deltaLon = lon1 - lon2 
    a= deltaLat.apply(lambda x: (math.sin(x/2))**2) + lat1.apply(lambda x: math.cos(x))*math.cos(lat2)*deltaLon.apply(lambda x: math.sin(x/2)**2)
    c = a.apply(lambda x: 2*math.atan2(math.sqrt(x),math.sqrt(1-x)))
    x = deltaLon* lat1.apply(lambda x: math.cos((x+lat2)/2))
    y = deltaLat
    d = x*x + y*y
    df_city['Hav_dist'] = round(radius*c,2)   #  Haversine distance
    df_city['Pyth_dist']  = round(d.apply(lambda i: math.sqrt(i)*radius),2)# Pythogorian distance 
    df_filtered = df_city[df_city['Hav_dist'] < r]
    df_filtered['Id'] = [x for x in range(1,df_filtered.shape[0]+1)] 
    
    return df_filtered




def getSocialMediaWithinRadius(df_post, lat, lon, r, gmaps):
    radius = 3958.8 # Earth radius
    lat1 = df_post['LATITUDE']*math.pi/180
    lat2 = lat*math.pi/180
    lon1 = df_post['LONGITUDE']*math.pi/180
    lon2 = lon*math.pi/180
    deltaLat = lat1 - lat2
    deltaLon = lon1 - lon2 
    a= deltaLat.apply(lambda x: (math.sin(x/2))**2) + lat1.apply(lambda x: math.cos(x))*math.cos(lat2)*deltaLon.apply(lambda x: math.sin(x/2)**2)
    c = a.apply(lambda x: 2*math.atan2(math.sqrt(x),math.sqrt(1-x)))
    x = deltaLon* lat1.apply(lambda x: math.cos((x+lat2)/2))
    y = deltaLat
    d = x*x + y*y
    df_post['Hav_dist'] = round(radius*c,2)   #  Haversine distance
    df_post['Pyth_dist']  = round(d.apply(lambda i: math.sqrt(i)*radius),2)# Pythogorian distance 
    df_post= df_post[df_post['Hav_dist'] < r]
    df_post['Id'] = [x for x in range(1,df_post.shape[0]+1)] 
    
    return df_post


#def getGasstationWithinDistance(df, city, Add, r):
#    df_city= df[df['CITY'] == city]
#    Add = df_city['ADDRESS'] + ',' + " " +  df_city['CITY']
#    Matrix1 = gmaps.distance_matrix(origins = [Add.to_numpy().tolist()[0]], 
#                                               destinations = Add.to_numpy().tolist())  
    
    
def getDistanceDurationSearch(df_filtered, lat, lon):
    Locations = df_filtered['Location'].to_numpy().tolist()
    Locations = [[lat, lon]] + Locations
    Matrix = gmaps.distance_matrix(origins = Locations, destinations = Locations)
    Distance = {(i,j):  Matrix['rows'][i]['elements'][j]['distance']['text'] for i in range(0,len(Locations)) 
                 for j in range(0,len(Locations)) if i != j}
    d = {key: float(str.split(Distance[key],sep=' ')[0]) for key in Distance}
    Duration = {(i,j):  Matrix['rows'][i]['elements'][j]['duration']['text'] for i in range(0,len(Locations)) 
                 for j in range(0,len(Locations)) if i != j}
    t = {key: float(str.split(Duration[key],sep=' ')[0]) for key in Duration}
    
    return d,t 
       

    
    
def getDistanceDurationBayes(df_filtered, df_post):
    Destinations = df_filtered['Location'].to_numpy().tolist()
    Latitudes  =  df_post['LATITUDE'].to_numpy().tolist()
    Longitudes = df_post['LONGITUDE'].to_numpy().tolist()
    Origins = [[Latitudes[p], Longitudes[p]] for p in range(len(Latitudes))]
    Matrix = gmaps.distance_matrix(origins = Origins, destinations = Destinations)
    Distance = {(i,p):  Matrix['rows'][p]['elements'][i]['distance']['text'] for p in range(0,len(Origins)) 
                 for i in range(0,len(Destinations)) if i != p}
    d = {key: float(str.split(Distance[key],sep=' ')[0]) for key in Distance}
    Duration = {(i,p):  Matrix['rows'][p]['elements'][i]['duration']['text'] for p in range(0,len(Origins)) 
                 for i in range(0,len(Destinations)) if i != p}
    t = {key: float(str.split(Duration[key],sep=' ')[0]) for key in Duration}
    
    return d,t 
       



