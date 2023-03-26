
import os

from django.conf.global_settings import MEDIA_ROOT
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import cv2

from django.shortcuts import render

from segmentation.forms import ImageMaskForm
from segmentation.utils import process_mask, colorize_segmentation, segment_image

SEGMENTATION_COLORS = {
    0: [0, 0, 0],           # void
    1: [128, 64, 128],      # flat	road · sidewalk · parking+ · rail track+
    2: [70, 70, 70],        # construction	building · wall · fence · guard rail+ · bridge+ · tunnel+
    3: [220, 220, 0],       # object	pole · pole group+ · traffic sign · traffic light
    4: [107, 142, 35],        # nature	vegetation · terrain
    5: [70, 130, 180],        # sky
    6: [220, 20, 60],       # human	person* · rider*
    7: [0, 0, 142]           # vehicle	car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+
}


def home(request):
    return render(request, 'segmentation/home.html')


def models(request):
    return render(request, 'segmentation/models.html')


def about(request):
    return render(request, 'segmentation/about.html')


@csrf_exempt
def segmentation(request):
    if request.method == 'POST':
        form = ImageMaskForm(request.POST, request.FILES)

        if form.is_valid():
            # Récupèration de l'image téléchargée à partir de la requête POST
            uploaded_image = form.cleaned_data['image']
            # Enregistrement de l'image dans le système de fichiers de Django
            fs = FileSystemStorage()

            # Suprression des fichiers précédents
            for directrory in fs.listdir(MEDIA_ROOT):
                for file in directrory:
                    fs.delete(os.path.join(MEDIA_ROOT, file))
            # Sauvegarde du nouveau fichier image
            image_name = fs.save(uploaded_image.name, uploaded_image)

            # Obtenez le chemin de l'image téléchargée
            uploaded_image_url = fs.url(image_name)
            uploaded_image_path = os.path.join(MEDIA_ROOT, uploaded_image_url[1:])

            # Si un masque est fournit, traitement du masque
            if form.cleaned_data['mask']:
                uploaded_mask = form.cleaned_data['mask']
                # Sauvegarde du masque
                mask_name = fs.save(uploaded_mask.name, uploaded_mask)
                # Récupération du chemin du mask
                uploaded_mask_url = fs.url(mask_name)
                uploaded_mask_path = os.path.join(MEDIA_ROOT, uploaded_mask_url[1:])
                mask_ohe = process_mask(uploaded_mask_path)
                mask_label = np.argmax(mask_ohe, axis=-1)
                colored_mask = colorize_segmentation(mask_label, colors=SEGMENTATION_COLORS)
                initial_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
                cv2.imwrite('media/initial_mask.png', initial_mask)

            # Chargez l'image avec OpenCV
            img = cv2.imread(uploaded_image_path)
            # Effectuez la segmentation avec votre modèle de segmentation
            # Remplacez cette ligne avec votre propre code de segmentation

            predict_mask = segment_image(img, model_name=form.cleaned_data['model'])
            # Recolore le masque
            colored_predict_mask = colorize_segmentation(predict_mask, colors=SEGMENTATION_COLORS)

            # Convertissez l'image segmentée en format compatible avec Django
            predict_mask = cv2.cvtColor(colored_predict_mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite('media/predict_mask.png', predict_mask)

            # Renvoyez l'URL de l'image segmentée en réponse
            return render(request,
                          'segmentation/results.html',
                          {
                              'initial_image':  uploaded_image_url,
                              'initial_mask': f'{settings.MEDIA_URL}initial_mask.png',
                              'predict_mask': f'{settings.MEDIA_URL}predict_mask.png',
                              'model_name': form.cleaned_data['model'],
                          })
        else:
            return render(request, 'segmentation/upload.html', {'form': form})

    else:
        form = ImageMaskForm()
        return render(request, 'segmentation/upload.html', {'form': form})


def handler404(request, exception):
    return render(request, '404.html', status=404)
