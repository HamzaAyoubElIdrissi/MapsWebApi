from django.contrib import admin
from .models import Address
import requests
from . import kmeans_utils

def geocode_address(address):
    api_key = 'AIzaSyCVn-TxhgR6GJ8IdftOTLZquU-gHWNPXz4'
    geocoding_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    response = requests.get(geocoding_url)
    data = response.json()

    if data['status'] == 'OK':
        result = data['results'][0]
        latitude = result['geometry']['location']['lat']
        longitude = result['geometry']['location']['lng']
        return latitude, longitude
    return None, None

@admin.register(Address)
class AddressAdmin(admin.ModelAdmin):
    list_display = ('id', 'address', 'latitude', 'longitude', 'cluster')

    def save_model(self, request, obj, form, change):
        if not obj.latitude or not obj.longitude or not obj.cluster:
            # If latitude, longitude, or cluster is missing, geocode the address and predict the cluster
            latitude, longitude = geocode_address(obj.address)
            if latitude is not None and longitude is not None:
                obj.latitude = latitude
                obj.longitude = longitude

                Map_kmeans = kmeans_utils.load_kmeans_model('map_app/models/Map_kmeans.pkl')
                user_cluster = kmeans_utils.predict_cluster(Map_kmeans, latitude, longitude)
                obj.cluster = user_cluster

        super().save_model(request, obj, form, change)