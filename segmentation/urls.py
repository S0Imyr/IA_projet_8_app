from django.urls import path

from . import views

urlpatterns = [

    path('', views.home, name='home'),
    path('models', views.models, name='models'),
    path('segmentate/', views.segmentation, name='segmentation'),
    path('about', views.about, name='about'),
]
