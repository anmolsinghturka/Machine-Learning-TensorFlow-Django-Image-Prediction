# -*- coding: utf-8 -*-
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ProductList.as_view(), name='product_list'),
    path('upload_image', views.upload_image, name='upload_image'),
]
