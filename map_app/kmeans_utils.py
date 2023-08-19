import mysql.connector
import pandas as pd
from sklearn.cluster import KMeans
import folium
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


# Global variable to store the trained KMeans model
Map_kmeans = None

def train_kmeans_model():
    # MySQL database connection
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="P@ssw0rd171295",
        database="addresses"
    )
    # Create a cursor object to interact with the database
    cursor = mydb.cursor()
    cursor1 = mydb.cursor()
    # SQL statement to select latitude and longitude from the addresses table
    select_sql = """
    SELECT latitude, longitude FROM addresses2
    """
    # Execute the select query
    cursor.execute(select_sql)
    # Fetch all rows from the result set
    result = cursor.fetchall()
    # Create a dataframe from the fetched rows
    df = pd.DataFrame(result, columns=['latitude', 'longitude'])
    #-------------------------------------------------------------
    # Select latitudes and longitudes from the addresses3 table
    select_query = "SELECT latitude, longitude FROM addresses3"
    cursor1.execute(select_query)
    result1 = cursor1.fetchall()
    # Close the cursor and database connection
    cursor1.close()
    mydb.close()
    # Create the DataFrame
    df_addr = pd.DataFrame(result1, columns=["latitude", "longitude"])
    df_addr.dropna(inplace=True)

    def frange(start, stop, step):
        while start < stop:
            yield start
            start += step

    # Define the polygon using the given coordinate points
    polygon_points = [(33.66379005277, -7.4546751941794005),
    (33.621222855361474, -7.532724556444049),
    (33.619249181418766, -7.5457808721910515),
    (33.61587054272837, -7.557787013211923),
    (33.61170390797176, -7.571128973711578),
    (33.607814866829855, -7.575798661658413),
    (33.599257797136296, -7.597281907396715),
    (33.60089512879704, -7.6170667058529595),
    (33.60364328133311, -7.617145359095142),
    (33.60560525796771, -7.619347648446281),
    (33.60749840845874, -7.617550422981526),
    (33.60785214131411, -7.6205431769836025),
    (33.609506150671024, -7.621805560723082),
    (33.610776932338396, -7.626084294265538),
    (33.608192771583276, -7.629902906865692),
    (33.60907709386139, -7.6321917148587906),
    (33.60911639686291, -7.633945680908165),
    (33.606813865776765, -7.6340754587036495),
    (33.60493380186395, -7.637575526111258),
    (33.603951173780445, -7.641437397547327),
    (33.605202384949585, -7.646608845276454),
    (33.60676146036787, -7.648936979786358),
    (33.60943409548396, -7.650647686688561),
    (33.613049881704654, -7.650981962761121),
    (33.61012844301615, -7.655425868139723),
    (33.608382737877605, -7.655937113943055),
    (33.601749372768936, -7.66210268357505),
    (33.59744650996894, -7.671219165106695),
    (33.596559005503124, -7.676851205123458),
    (33.59678175461015, -7.678447864800693),
    (33.597092947252285, -7.6791754067718445),
    (33.59414121437515, -7.679198342735542),
    (33.59097465697027, -7.679880663899287),
    (33.58557693216441, -7.69042490523074),
    (33.581243168211046, -7.703820038112942),
    (33.57811741657166, -7.704935146797044),
    (33.57658836985204, -7.708385012207812),
    (33.57656136169409, -7.708381075690673),
    (33.57448065280562, -7.710564828735482),
    (33.56870279334856, -7.720658879867773),
    (33.564705488446535, -7.728894417400402),
    (33.560546086467596, -7.738952480762015),
    (33.54793659969397, -7.758978763399179),
    (33.53603812527798, -7.7866365689098584),
    (33.53581543576626, -7.7945584815511335),
    (33.532842959961954, -7.794291320274376),
    (33.529248140427526, -7.802052803729091),
    (33.527073344183044, -7.809994328560902),
    (33.52717476075489, -7.816699003274818),
    (33.527891610121465, -7.8177128151022615),
    (33.53443120988151, -7.81908983692865),
    (33.53677532170333, -7.82307544207615),
    (33.53438271032027, -7.826750733289989),
    (33.53240228934455, -7.830183590278346),
    (33.5293790285447, -7.831735164258544),
    (33.51573804143036, -7.857600532141901),
    (33.51351091518839, -7.869066281738301),
    (33.49891325340347, -7.866929296150443),
    (33.37337773787243, -7.768401487880836),
    (33.32885280498702, -7.478525830956036),
    (33.64939484213151, -7.417938912891625)]

    polygon = Polygon(polygon_points)

    # Define the bounding box of the polygon
    min_lat = min(point[0] for point in polygon_points)
    max_lat = max(point[0] for point in polygon_points)
    min_lon = min(point[1] for point in polygon_points)
    max_lon = max(point[1] for point in polygon_points)

    # Define the grid resolution
    resolution = 0.0025  # Adjust as needed

    # Iterate over the grid and check if each point falls within the polygon
    points_inside_zone = []
    for lat in frange(min_lat, max_lat, resolution):
        for lon in frange(min_lon, max_lon, resolution):
            point = Point(lat, lon)
            if polygon.contains(point):
                points_inside_zone.append((lat, lon))

    # Create a DataFrame to store the points
    df_pol = pd.DataFrame(points_inside_zone, columns=['latitude', 'longitude'])

    df = pd.concat([df, df_addr])
    df = pd.concat([df, df_pol])
    df.drop_duplicates(inplace=True)

    # Calculate the interquartile range (IQR)
    Q1 = df['latitude'].quantile(0.25)
    Q3 = df['latitude'].quantile(0.75)
    IQR = Q3 - Q1

    LQ1 = df['longitude'].quantile(0.25)
    LQ3 = df['longitude'].quantile(0.75)
    LIQR = LQ3 - LQ1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    Llower_bound = LQ1 - 1.5 * LIQR
    Lupper_bound = LQ3 + 1.5 * LIQR
    df = df[(df['latitude'] >= lower_bound) & (df['latitude'] <= upper_bound)]
    df = df[(df['longitude'] >= Llower_bound) & (df['longitude'] <= Lupper_bound)]
    df.reset_index(drop=True, inplace=True)


    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    optimal_k = 6   # Select the optimal number of clusters based on the elbow curve
    Map_kmeans = KMeans(n_clusters=optimal_k)
    Map_kmeans.fit(train_df)
    joblib.dump(Map_kmeans, 'map_app/models/Map_kmeans.pkl')
    

def load_kmeans_model(model_path):
    # Load the pre-trained KMeans model
    kmeans_model = joblib.load(model_path)

    return kmeans_model

def predict_cluster(model, latitude, longitude):
    model_path = 'map_app/models/Map_kmeans.pkl'
    model = joblib.load(model_path)
    user_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
    # Use the KMeans model to predict the cluster for the user's address
    user_cluster = model.predict(user_df)[0]
    return int(user_cluster)


# def create_addresses_table():
#     mydb = mysql.connector.connect(
#         host="127.0.0.1",
#         user="root",
#         password="P@ssw0rd171295",
#         database="addresses"
#     )
#     cursor = mydb.cursor()

#     # Create the addresses4 table if it doesn't exist
#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS addresses4 (
#         id INT AUTO_INCREMENT PRIMARY KEY,
#         address VARCHAR(255) UNIQUE,
#         latitude FLOAT,
#         longitude FLOAT,
#         cluster INT
#     )
#     """
#     cursor.execute(create_table_query)

#     # Commit the changes and close the connection
#     mydb.commit()
#     cursor.close()
#     mydb.close()

# def insert_address_data(address, latitude, longitude, cluster):
#     mydb = mysql.connector.connect(
#         host="127.0.0.1",
#         user="root",
#         password="P@ssw0rd171295",
#         database="addresses"
#     )
#     cursor = mydb.cursor()

#     # Insert the address data into the addresses4 table if it's not a duplicate
#     insert_query = """
#     INSERT IGNORE INTO addresses4 (address, latitude, longitude, cluster)
#     VALUES (%s, %s, %s, %s)
#     """
#     data = (address, latitude, longitude, cluster)
#     cursor.execute(insert_query, data)

#     # Commit the changes and close the connection
#     mydb.commit()
#     cursor.close()
#     mydb.close()