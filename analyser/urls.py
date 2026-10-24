from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyse/', views.analyse_fichier, name='analyse_fichier'),
    path('api/domaine/', views.api_domaine_selected, name='api_domaine_selected'),
    path('api/analyse/', views.api_analyse_multidomaine, name='api_analyse_multidomaine'),
    path('analyser/', views.analyser_view, name='analyser_view'),
]