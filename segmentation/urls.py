from django.urls import path

from . import views

urlpatterns = [
    path('segmentation/', views.segmentation, name='segmentation'),
]
