# -*- coding: utf-8 -*-
import os
from io import BytesIO
import tensorflow as tf
import numpy as np
import pathlib
import datetime
from django.http import JsonResponse
from django.views.generic import ListView, DetailView
from .forms import ImageUploadForm
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.contrib.messages.views import SuccessMessageMixin
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.http import HttpResponse, JsonResponse
from django.urls import reverse_lazy
from .models import Product
from .forms import ProductForm


class ProductList(ListView):
    model = Product


class ProductDetail(DetailView):
    model = Product


class ProductCreate(SuccessMessageMixin, CreateView):
    model = Product
    form_class = ProductForm
    success_url = reverse_lazy('product_list')
    success_message = "Product successfully created!"


class ProductUpdate(SuccessMessageMixin, UpdateView):
    model = Product
    form_class = ProductForm
    success_url = reverse_lazy('product_list')
    success_message = "Product successfully updated!"


class ProductDelete(SuccessMessageMixin, DeleteView):
    model = Product
    success_url = reverse_lazy('product_list')
    success_message = "Product successfully deleted!"


batch_size = 32
img_height = 180
img_width = 180

# Load the modified model using the interpreter
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_class(image_path):
    # Load and preprocess the image.
    img = tf.keras.preprocessing.image.load_img(
        BytesIO(image_path.read()), target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Set the tensor data for the input tensor.
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Perform inference.
    interpreter.invoke()

    # Get the output tensor data.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Compute softmax activation.
    score_lite = tf.nn.softmax(output_data)

    # Get the predicted class name and confidence value.
    class_names = ['Bed', 'Chair', 'Sofa']
    class_name = class_names[np.argmax(score_lite)]
    confidence = 100 * np.max(score_lite)

    return class_name, confidence


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            print(
                f"Received image file: {image_file.name} at {datetime.datetime.now()}")

            # Get the predicted class name and confidence value.
            class_name, confidence = predict_class(image_file)

            # Return the results.
            return render(request, 'products/product_list.html', {'class_name': class_name, 'confidence': confidence})
    else:
        form = ImageUploadForm()
    return render(request, 'product_list.html', {'form': form})


def index(request):
    return render(request, 'index.html')
