"""
URL configuration for map_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.contrib import admin
from map_app.views import map_view, show_route,cluster_addresses_route, cluster_addresses_list, train_kmeans, cluster_list, cluster_directions, addresses_add,addresses_list

urlpatterns = [
    path("admin/", admin.site.urls),
    path('map/', map_view, name='map'),
    path('train_kmeans/', train_kmeans, name='train_kmeans'),
    path('addresses_add/', addresses_add, name='addresses_add'),
    path('addresses_list/', addresses_list, name='addresses_list'),
    path('cluster_list/', cluster_list, name='cluster_list'),
    path('cluster_directions/<int:cluster>/', cluster_directions, name='cluster_directions'),
    path('cluster_addresses_list/<int:cluster>/', cluster_addresses_list, name='cluster_addresses_list'),
    #path('optimized_routes/<int:cluster_id>/', optimized_routes, name='optimized_routes'),
    path('show_route/<int:cluster>/<str:user_lat>/<str:user_lng>/', show_route, name='show_route'),
    path('cluster_addresses_route/<int:cluster>/', cluster_addresses_route, name='cluster_addresses_route'),
   ]