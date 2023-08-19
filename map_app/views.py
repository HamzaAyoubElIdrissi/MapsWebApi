# map_app/views.py
import requests
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from .serializers import AddressSerializer
from .models import Address
from decimal import Decimal, ROUND_HALF_UP
import json
from django.db.models import Count
import pandas as pd
from django.shortcuts import render
import mysql.connector
from django.http import JsonResponse
import joblib
from django.views.decorators.csrf import csrf_exempt
from . import kmeans_utils
from rest_framework import status
from rest_framework.permissions import IsAdminUser
import googlemaps
from datetime import datetime
from googlemaps import Client

Map_kmeans = 'map_app/models/Map_kmeans.pkl'
google_maps_client = Client(key='AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4')
@csrf_exempt
def train_kmeans(request):
    kmeans_utils.train_kmeans_model()
    global Map_kmeans
    model_path = 'map_app/models/Map_kmeans.pkl'
    Map_kmeans = kmeans_utils.load_kmeans_model(model_path)
    return JsonResponse({"message": "KMeans model trained and saved successfully."})

@api_view(['GET'])
def addresses_list(request):
    if request.method == 'GET':
        addresses = Address.objects.all()
        serializer = AddressSerializer(addresses, many=True)
        return Response({'addresses': serializer.data})

    return Response(status=400)

@api_view(['GET'])
def cluster_addresses_list(request, cluster):
    if request.method == 'GET':
        addresses = Address.objects.filter(cluster=cluster)
        serializer = AddressSerializer(addresses, many=True)
        return Response({'addresses': serializer.data})

    return Response(status=400)


@api_view(['GET'])
def cluster_addresses_route(request, cluster):
    if request.method == 'GET':
        addresses = Address.objects.filter(cluster=cluster).values('latitude', 'longitude')
        return JsonResponse({'addresses': list(addresses)})

    return Response(status=400)


@api_view(['GET'])
def cluster_list(request):
    if request.method == 'GET':
        addresses = Address.objects.all()
        serializer = AddressSerializer(addresses, many=True)

        # Get unique clusters from the addresses
        clusters = addresses.values('cluster').annotate(count=Count('cluster'))

        # Prepare the response data with unique clusters
        response_data = {
            'clusters': [{'id': cluster['cluster']} for cluster in clusters]
        }
        return Response(response_data)

    return Response(status=400)



@csrf_exempt
@api_view(['POST'])
def addresses_add(request):
    if request.method == 'POST':
        try:
            address = request.data.get('address', '')
            print(address)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

        if address:
            # Use the geocoding API to get latitude and longitude
            api_key = 'AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4'
            geocoding_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
            response = requests.get(geocoding_url)
            data = response.json()

            if data['status'] == 'OK':
                result = data['results'][0]
                latitude = result['geometry']['location']['lat']
                longitude = result['geometry']['location']['lng']
                
                latitude = Decimal(latitude).quantize(Decimal('0.000000000000001'), rounding=ROUND_HALF_UP)
                longitude = Decimal(longitude).quantize(Decimal('0.000000000000001'), rounding=ROUND_HALF_UP)
                print(str(latitude) + ',' + str(longitude))
                print(str(latitude) + ',' + str(longitude))
                
                global Map_kmeans

                if Map_kmeans is None:
                    model_path = 'map_app/models/Map_kmeans.pkl'
                    Map_kmeans = kmeans_utils.load_kmeans_model(model_path)

                user_cluster = kmeans_utils.predict_cluster(Map_kmeans, latitude, longitude)
                print(user_cluster)

                # Check if the address with the same coordinates exists in the database
                existing_address = Address.objects.filter(address=address,latitude=latitude, longitude=longitude).first()
                if existing_address:
                    return Response({'error': 'Address with the same coordinates already exists'}, status=400)

                # Save the address data in the database
                serializer = AddressSerializer(data={
                    'address': address,
                    'latitude': latitude,
                    'longitude': longitude,
                    'cluster': user_cluster
                })

                # Debug: Print serializer errors
                if not serializer.is_valid():
                    print(serializer.errors)

                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=201)
                return Response(serializer.errors, status=400)

    return Response(status=400)




@permission_classes([IsAdminUser])
@api_view(['GET', 'PUT', 'DELETE'])
def address_detail(request, pk):
    try:
        address = Address.objects.get(pk=pk)
    except Address.DoesNotExist:
        return Response({'error': 'Address not found'}, status=404)

    if request.method == 'GET':
        serializer = AddressSerializer(address)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = AddressSerializer(address, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        address.delete()
        return Response(status=204)





@csrf_exempt
def kmeans_endpoint(request):
    if request.method == 'POST':
         # Parse the JSON data from the request body
        try:
            data = json.loads(request.body)
            address = data.get('address', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

        #address = request.POST.get('address', '')  # Get the address from the user
        if address:
            # Use the geocoding API to get latitude and longitude
            api_key = 'AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4'
            geocoding_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
            response = requests.get(geocoding_url)
            data = response.json()

            if data['status'] == 'OK':
                result = data['results'][0]
                latitude = result['geometry']['location']['lat']
                longitude = result['geometry']['location']['lng']
                print("Latitude:", latitude)  # Debugging statement
                print("Longitude:", longitude)  # Debugging statement
                

                global Map_kmeans

                if Map_kmeans is None:
                    model_path = 'map_app/models/Map_kmeans.pkl'
                    Map_kmeans = kmeans_utils.load_kmeans_model(model_path)

                # Store the address data in the database
                kmeans_utils.create_addresses_table()  # Create the addresses4 table if it doesn't exist

                # Use the KMeans model to predict the cluster for the user's address
                user_cluster = kmeans_utils.predict_cluster(Map_kmeans, latitude, longitude)

                kmeans_utils.insert_address_data(address, latitude, longitude, user_cluster)

                # Use the KMeans model to predict the cluster for the user's address
                
                # Print the cluster prediction for debugging
                print("Cluster Prediction:", user_cluster)




                
                # Return the cluster prediction as a JSON response
                return JsonResponse({'latitude': latitude, 'longitude': longitude, 'cluster': int(user_cluster)})
    
    
    # If the request is not a POST request or there's no address provided, return an empty JSON response
    return JsonResponse({})

def map_view(request):
    # If the request is not a POST request or there's no address provided, render the map template
    api_key = 'AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4'  # Replace with your actual API key
    return render(request, 'map.html', {'api_key': api_key})


def addresses_table(request):
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="P@ssw0rd171295",
        database="address"
    )
    cursor = mydb.cursor(dictionary=True)
    select_query = "SELECT * FROM map_app_address"
    cursor.execute(select_query)
    addresses = cursor.fetchall()
    cursor.close()
    mydb.close()
    return JsonResponse({'addresses': addresses})


# Add a new view to calculate and return optimized routes between addresses of the same cluster
@csrf_exempt
def cluster_directions(request,cluster):
    if request.method == 'GET':
        # # Get the selected cluster from the request
        # cluster = request.POST.get('cluster', None)
        # if cluster is None:
        #     return JsonResponse({'error': 'Cluster not provided'}, status=400)

        # Get all addresses belonging to the selected cluster from the database
        addresses = Address.objects.filter(cluster=cluster)
        if not addresses.exists():
            return JsonResponse({'error': 'No addresses found for the selected cluster'}, status=404)

        # Extract latitude and longitude pairs from addresses
        waypoints = [(address.latitude, address.longitude) for address in addresses]

        # Calculate optimized directions using Google Maps Directions API
        api_key = 'AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4'
        gmaps = googlemaps.Client(key=api_key)

        directions = gmaps.directions(
            origin=waypoints[0],
            destination=waypoints[-1],
            waypoints=waypoints[1:(len(waypoints) - 1)],
            optimize_waypoints=True
        )

        if not directions:
            return JsonResponse({'error': 'Failed to calculate directions'}, status=500)
        # Request parameters for the Google Maps Directions API
        directions_request_params = {
            'origin': waypoints[0],
            'destination': waypoints[-1],
            'waypoints': waypoints[1:(len(waypoints) - 1)],
        }
        # Extract the route information from the directions response (you can further process this data as needed)
        route_info = []
        route_coordinates = []  # To store the latitude and longitude pairs for the route
        total_distance = 0  # To store the total distance of the entire route
        total_duration = 0  # To store the total duration of the entire route

        for leg in directions[0]['legs']:
            for step in leg['steps']:
                route_info.append(step['html_instructions'])

                # Extract latitude and longitude of each step and store in route_coordinates
                start_lat = step['start_location']['lat']
                start_lng = step['start_location']['lng']
                route_coordinates.append({'lat': start_lat, 'lng': start_lng})

                end_lat = step['end_location']['lat']
                end_lng = step['end_location']['lng']
                route_coordinates.append({'lat': end_lat, 'lng': end_lng})

                # Use the Distance Matrix API to calculate distance and duration between each step
                origins = f"{start_lat},{start_lng}"
                destinations = f"{end_lat},{end_lng}"
                distance_matrix = gmaps.distance_matrix(origins, destinations, mode='driving')

                if distance_matrix['status'] == 'OK':
                    step_distance = distance_matrix['rows'][0]['elements'][0]['distance']['value']
                    step_duration = distance_matrix['rows'][0]['elements'][0]['duration']['value']
                    total_distance += step_distance
                    total_duration += step_duration

        # Convert total_distance and total_duration from meters and seconds to human-readable format
        total_distance_km = total_distance / 1000
        total_duration_min = total_duration / 60

        return JsonResponse({
            'directions': route_info,
            'coordinates': route_coordinates,
            'total_distance_km': total_distance_km,
            'total_duration_min': total_duration_min,
        })
    return JsonResponse({'error': 'Invalid request method'}, status=400)

# @csrf_exempt
# @api_view(['GET'])
# def show_route(request, cluster):
#     # Convert rest_framework.request.Request to django.http.HttpRequest
#     django_request = request._request

#     # Call the directions view
#     #
#     # Print the keys in directions_data_dict
#     #print("Keys in directions_data_dict:", directions_data_dict.keys())

#     # Call the addresses route view
#     addresses_data = cluster_addresses_route(django_request, cluster)
    
#     addresses_data_content = addresses_data.content
    
#     addresses_data_dict = json.loads(addresses_data_content)
    
#     if addresses_data_dict.get('addresses') is None:
#         print("Failed to fetch addresses data")
#         return JsonResponse({'error': 'Failed to fetch addresses data'})

#     # Print the keys in addresses_data_dict
#     #print("Keys in addresses_data_dict:", addresses_data_dict.keys())

#     # Convert directions coordinates format to match addresses format
#     #converted_directions_coordinates = [{'latitude': coord['lat'], 'longitude': coord['lng']} for coord in directions_data_dict.get('coordinates', [])]
#     # Add index ID to directions
#     #directions = pd.DataFrame(converted_directions_coordinates)
#     # Reset index for the directions DataFrame
#     #directions.reset_index(drop=True, inplace=True)
#     addresses_df = pd.DataFrame(addresses_data_dict['addresses'])

#     # Convert latitude and longitude columns to appropriate data types
#     # directions['latitude'] = directions['latitude'].astype(float)
#     # directions['longitude'] = directions['longitude'].astype(float)
#     # addresses_df['latitude'] = addresses_df['latitude'].astype(float)
#     # addresses_df['longitude'] = addresses_df['longitude'].astype(float)
    
#     # # Quantize the coordinates to 16 decimals
#     # directions['latitude'] = directions['latitude'].apply(lambda x: round(x, 16))
#     # directions['longitude'] = directions['longitude'].apply(lambda x: round(x, 16))
#     # addresses_df['latitude'] = addresses_df['latitude'].apply(lambda x: round(x, 16))
#     # addresses_df['longitude'] = addresses_df['longitude'].apply(lambda x: round(x, 16))
    
    
#     # merged_data = directions.merge(addresses_df, on=['latitude', 'longitude'], how='inner')

#     # Convert merged_data DataFrame to dictionary
#     #merged_data_dict = merged_data.to_dict(orient='records')

#     # Convert latitude and longitude columns to appropriate data types
#     # addresses_df['latitude'] = addresses_df['latitude'].astype(float)
#     # addresses_df['longitude'] = addresses_df['longitude'].astype(float)
    
#     # Call Google Distance Matrix API for distances between addresses
#     origin_coords = addresses_df[['latitude', 'longitude']].values.tolist()
#     destination_coords = addresses_df[['latitude', 'longitude']].values.tolist()
#     matrix = google_maps_client.distance_matrix(origin_coords, destination_coords, mode="driving")

#     # Process the distance matrix response to get optimal route and details
#     routes = matrix['rows']

#     # Find the optimal route based on the shortest distance
#     optimal_route_index = min(range(len(routes)), key=lambda i: routes[i]['elements'][i]['distance']['value'])
#     optimal_route = routes[optimal_route_index]['elements']
    
#     # Extract the optimal route coordinates
#     optimal_route_coords = []
#     for i, element in enumerate(optimal_route):
#         coords = destination_coords[i]
#         optimal_route_coords.append({'latitude': coords[0], 'longitude': coords[1]})
    
#     # Calculate total distance and total duration
#     total_distance = sum(element['distance']['value'] for element in optimal_route)
#     total_duration = sum(element['duration']['value'] for element in optimal_route)
    
#     return JsonResponse({
#         'optimal_route_coordinates': optimal_route_coords,
#         'total_distance': total_distance,
#         'total_duration': total_duration
#     })


@csrf_exempt
@api_view(['GET'])
def show_route(request, cluster, user_lat, user_lng):
    # Convert rest_framework.request.Request to django.http.HttpRequest
    django_request = request._request

    # Call the addresses route view
    addresses_data = cluster_addresses_route(django_request, cluster)
    
    addresses_data_content = addresses_data.content
    
    addresses_data_dict = json.loads(addresses_data_content)
    
    if addresses_data_dict.get('addresses') is None:
        print("Failed to fetch addresses data")
        return JsonResponse({'error': 'Failed to fetch addresses data'})

    addresses_df = pd.DataFrame(addresses_data_dict['addresses'])

    # Convert latitude and longitude columns to appropriate data types
    #addresses_df['latitude'] = addresses_df['latitude'].astype(float)
    #addresses_df['longitude'] = addresses_df['longitude'].astype(float)
    # Define the user's location as the origin
    user_location = (user_lat, user_lng)
    # Add user's location to the top of addresses_df
    #user_row = {'latitude': user_lat, 'longitude': user_lng}
    #addresses_df = pd.concat([pd.DataFrame([user_row]), addresses_df], ignore_index=True)

    print(addresses_df)
    # Shuffle the coordinates before sending them to the API
    shuffled_coords = addresses_df.sample(frac=1).reset_index(drop=True)
    # origin_coords = [(user_lat, user_lng)] + addresses_df[['latitude', 'longitude']].values.tolist()
    destination_coords = [user_location] + shuffled_coords[['latitude', 'longitude']].values.tolist()
    
    # Call Google Distance Matrix API for distances between addresses
    matrix = google_maps_client.distance_matrix([destination_coords[0]], destination_coords[1:], mode="driving")

    # Process the distance matrix response to get optimal route and details
    routes = matrix['rows']

    # Find the optimal route based on the shortest duration
    optimal_route_index = min(range(len(routes)), key=lambda i: sum(element['duration']['value'] for element in routes[i]['elements']))
    optimal_route = routes[optimal_route_index]['elements']
    
    # Extract the optimal route coordinates
    optimal_route_coords = []
    for i, element in enumerate(optimal_route):
        coords = destination_coords[i]
        optimal_route_coords.append({'latitude': coords[0], 'longitude': coords[1]})
    
    # Calculate total distance and total duration
    total_distance = sum(element['distance']['value'] for element in optimal_route)
    total_duration = sum(element['duration']['value'] for element in optimal_route)
    
    return JsonResponse({
        'optimal_route_coordinates': optimal_route_coords,
        'total_distance': total_distance,
        'total_duration': total_duration
    })


# @api_view(['GET'])
# def optimized_routes(request, cluster_id):
#     if request.method == 'GET':
#         # Fetch all addresses belonging to the given cluster ID
#         addresses = Address.objects.filter(cluster=cluster_id)
        
#         # Get their latitude and longitude
#         waypoints = [(address.latitude, address.longitude) for address in addresses]
        
#         # Add the first address as the start and end location
#         start_location = waypoints[0]
#         end_location = waypoints[0]
        
#         # Calculate the optimized route using Google Maps Directions API
#         gmaps = googlemaps.Client(key='AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4')
#         now = datetime.now()
#         directions_result = gmaps.directions(start_location, end_location,
#                                              waypoints=waypoints,
#                                              mode="driving",
#                                              departure_time=now)
        
#         # Extract the optimized route details
#         route_summary = directions_result[0]['summary']
#         route_distance = directions_result[0]['legs'][0]['distance']['text']
#         route_duration = directions_result[0]['legs'][0]['duration']['text']
        
#         # Return the route details as a JSON response
#         return JsonResponse({
#             'cluster_id': cluster_id,
#             'route_summary': route_summary,
#             'route_distance': route_distance,
#             'route_duration': route_duration,
#             'waypoints': waypoints,
#         })

#     return Response(status=400)


