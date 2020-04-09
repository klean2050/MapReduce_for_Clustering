import numpy as np
from functools import partial
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkSession.builder.appName("AdvDat").getOrCreate()

def haversine(p,q):

    """Calculate haversine distance based on coordinates p,q"""

    lng1, lat1 = p
    lng2, lat2 = q
    dlng = abs(lng2-lng1)
    dlat = abs(lat2-lat1)

    a = np.sin(0.5*dlng)**2 + np.cos(lng1)*np.cos(lng2)*np.sin(0.5*dlat)**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return c*6371 # earth radius


def coordinates(line):

    """Produces coordinates tuples
    line: CSV string from RDD"""

    contents = line.split(",")
    lng, lat = map(float,contents[3:5])
    return lng, lat


def closest(centroids,coordinates):

    """Computes the minimum haversine distance and saves centroid
    centroids: contains the centroids as tuples (id,coordinates)
    coordinates: tuple of floats
        
    Returns id_coords: tuple (int,tuple)
    Tuple containing the id of the closest centroid
    and the tuple of coordinates of this point"""
    
    tup = [(cen[0], haversine(coordinates,cen[1])) for cen in centroids]
    distance = min(tup, key = lambda x:x[1])
    return (distance[0],coordinates)


def sum_by_elem(p,q):

    """Reduce Function, sums each coordinate of 2 items
    p,q: tuples of (tuple of floats: coordinates,int)
    Returns tuple of (tuple of summed floats, summed int)"""

    p, num1 = p
    q, num2 = q
    tup = map(sum,zip(p,q))
    return (tuple(tup),num1+num2)

def avg_by_elem(p):

    """Averages the coordinates assigned to centroid
    p: tuple returned by sum_by_elem
    Returns tuple of floats: mean of coordinates"""

    p, num = p
    avg = map(lambda x:x/num, p)
    return tuple(avg)


data = sc.textFile("yellow_tripdata_1m.csv
# get all coordinates as tuples and filter outliers
population = data.map(coordinates).filter(lambda line: line[0] and line[1]).cache()

# take first 5 as initial centroids with id >= 1
centroids = list(enumerate(population.take(5),1))

for _ in range(3):
    # assign current centroids to closest() function
    p_closest = partial(closest,centroids)
    # find closest centroid for each pair of coordinates
    cen_coords = population.map(p_closest)
    # find new centroids by map-reduce-map
    new_centroids = cen_coords.mapValues(lambda label:(label,1))\
                              .reduceByKey(sum_by_elem)\
                              .mapValues(avg_by_elem)
    centroids = new_centroids.collect()

# save results in csv file
centroids = new_centroids.toDF(["id","centroid"])
centroids.withColumn("centroid", centroids.centroid.cast("string"))\
         .coalesce(1).write\
         .csv("/Centroids",header="true",mode="overwrite",quote="")


