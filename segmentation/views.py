
import os

from django.conf.global_settings import MEDIA_ROOT
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render, redirect
from keras.utils import to_categorical

from segmentation_models import FPN
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore
from segmentation_models.metrics import FScore

from segmentation.forms import ImageMaskForm

IMG_SHAPE = (256, 256, 3)
NUM_CLASSES = 8
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


CATS = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
}


# Chargement du modèle
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
dice_loss = DiceLoss()
iou_score = IOUScore()
f_score = FScore()
metrics = [iou_score, f_score, 'accuracy']
model = FPN(input_shape=IMG_SHAPE, classes=NUM_CLASSES)
MODEL_WEIGHTS = 'segmentation/tfmodels/weights_fpn_vgg16_256p.h5'

model.load_weights(MODEL_WEIGHTS)
model.compile(optimizer=optimizer, loss=dice_loss, metrics=metrics)


def colorize_segmentation(segmented_img, colors):
    """
    Convertit une image segmentée en niveaux de gris en une image en couleurs.

    Args:
        segmented_img (numpy.ndarray): Image segmentée en niveaux de gris.
        colors (List[Tuple[int]]): Liste de tuples de trois entiers représentant les couleurs
            à utiliser pour chaque classe de segmentation. La longueur de cette liste doit être
            égale au nombre de classes.

    Returns:
        numpy.ndarray: Image segmentée en couleurs.
    """
    # Vérifiez que la longueur de la liste des couleurs correspond au nombre de classes
    n_classes = np.max(segmented_img) + 1
    assert len(colors) == n_classes, f"La liste des couleurs doit contenir {n_classes} couleurs."

    # Initialisez l'image segmentée en couleurs
    height, width = segmented_img.shape
    colorized_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Appliquez les couleurs correspondantes à chaque classe
    for i in range(n_classes):
        colorized_img[segmented_img == i] = colors[i]

    return colorized_img


def preprocess(image):
    # Prétraitement de l'image
    image = cv2.resize(image, (256, 256))  # Redimensionnement de l'image
    image = image / 255.0  # Normalisation des valeurs de pixels
    image = np.expand_dims(image, axis=0)  # Ajout d'une dimension pour correspondre à la forme d'entrée du modèle
    return image


# Fonction pour effectuer la segmentation de l'image
def segment_image(image):
    # Prétraitement de l'image
    image = preprocess(image)

    # Prédiction de la segmentation de l'image
    prediction = model.predict(image)

    # Conversion de la prédiction en image segmentée
    prediction = np.squeeze(prediction, axis=0)  # Suppression de la dimension ajoutée précédemment
    prediction = np.argmax(prediction, axis=-1)  # Extraction des classes avec la valeur maximale
    prediction = prediction.astype(np.uint8)  # Conversion en entiers non signés de 8 bits
    prediction = cv2.resize(prediction, (image.shape[1], image.shape[2]),
                            interpolation=cv2.INTER_NEAREST)  # Redimensionnement de la prédiction à la taille de l'image d'origine
    return prediction


def convert_mask(img, one_hot_encoder=False):
    if len(img.shape) == 3:
        img = np.squeeze(img[:, :, 0])
    else:
        img = np.squeeze(img)
    mask = np.zeros((img.shape[0], img.shape[1], 8), dtype=np.uint16)
    for i in range(-1, 34):
        if i in CATS['void']:
            mask[:, :, 0] = np.logical_or(mask[:, :, 0], (img == i))
        elif i in CATS['flat']:
            mask[:, :, 1] = np.logical_or(mask[:, :, 1], (img == i))
        elif i in CATS['construction']:
            mask[:, :, 2] = np.logical_or(mask[:, :, 2], (img == i))
        elif i in CATS['object']:
            mask[:, :, 3] = np.logical_or(mask[:, :, 3], (img == i))
        elif i in CATS['nature']:
            mask[:, :, 4] = np.logical_or(mask[:, :, 4], (img == i))
        elif i in CATS['sky']:
            mask[:, :, 5] = np.logical_or(mask[:, :, 5], (img == i))
        elif i in CATS['human']:
            mask[:, :, 6] = np.logical_or(mask[:, :, 6], (img == i))
        elif i in CATS['vehicle']:
            mask[:, :, 7] = np.logical_or(mask[:, :, 7], (img == i))

    if one_hot_encoder:
        return np.array(mask, dtype='uint8')
    else:
        return np.array(np.argmax(mask, axis=2), dtype='uint8')


def process_mask(mask_path, n_classes=8, one_hot_encoder: bool = False):
    mask = convert_mask(cv2.imread(mask_path, 0), one_hot_encoder=one_hot_encoder)
    mask_ = cv2.resize(mask, dsize=(256, 256))
    train_masks_cat = to_categorical(mask_, num_classes=n_classes)
    y = train_masks_cat.reshape((mask_.shape[0], mask_.shape[1], n_classes))
    return y


@csrf_exempt
def segmentation(request):
    if request.method == 'POST':
        form = ImageMaskForm(request.POST, request.FILES)

        if form.is_valid():
            # Obtenez l'image téléchargée à partir de la requête POST
            uploaded_image = form.cleaned_data['image']
            # Enregistrez l'image dans le système de fichiers de Django
            fs = FileSystemStorage()
            for directrory in fs.listdir(MEDIA_ROOT):
                for file in directrory:
                    fs.delete(os.path.join(MEDIA_ROOT, file))
            image_name = fs.save(uploaded_image.name, uploaded_image)
            # Obtenez le chemin de l'image téléchargée
            uploaded_image_url = fs.url(image_name)
            uploaded_image_path = os.path.join(MEDIA_ROOT, uploaded_image_url[1:])

            if form.cleaned_data['mask']:
                uploaded_mask = form.cleaned_data['mask']
                mask_name = fs.save(uploaded_mask.name, uploaded_mask)
                # Obtenez le chemin de l'image téléchargée
                uploaded_mask_url = fs.url(mask_name)
                uploaded_mask_path = os.path.join(MEDIA_ROOT, uploaded_mask_url[1:])
                mask = process_mask(uploaded_mask_path)
                mask_ohe = np.argmax(mask, axis=-1)

                colored_mask = colorize_segmentation(mask_ohe, colors=SEGMENTATION_COLORS)
                initial_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
                cv2.imwrite('media/initial_mask.png', initial_mask)

            # Chargez l'image avec OpenCV
            img = cv2.imread(uploaded_image_path)
            # Effectuez la segmentation avec votre modèle de segmentation
            # Remplacez cette ligne avec votre propre code de segmentation
            predict_mask = segment_image(img)
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
                          })
        else:
            return render(request, 'segmentation/upload.html', {'form': form})

    else:
        form = ImageMaskForm()
        return render(request, 'segmentation/upload.html', {'form': form})

