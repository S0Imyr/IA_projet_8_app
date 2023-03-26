
import numpy as np
import cv2

import tensorflow as tf
from keras.utils import to_categorical

from segmentation_models import FPN, Unet
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore
from segmentation_models.metrics import FScore


IMG_SHAPE = (256, 256, 3)
NUM_CLASSES = 8
CATS = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}


# Chargement du modèle
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
dice_loss = DiceLoss()
iou_score = IOUScore()
f_score = FScore()
metrics = [iou_score, f_score, 'accuracy']
model = FPN(input_shape=IMG_SHAPE, classes=NUM_CLASSES)
MODEL_WEIGHTS = {
    'FPN': 'segmentation/tfmodels/weights_fpn_vgg16_256p.h5',
    'FPN_AUG': 'segmentation/tfmodels/weights_fpn_vgg16_256p_aug.h5',
    'UNET': 'segmentation/tfmodels/weights_unet_vgg16_256p.h5',
    'UNET_AUG': 'segmentation/tfmodels/weights_unet_vgg16_256p_aug.h5',
}


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
def segment_image(image, model_name):
    # Prétraitement de l'image
    image = preprocess(image)
    if model_name in ["FPN", "FPN_AUG"]:
        model = FPN(input_shape=IMG_SHAPE, classes=NUM_CLASSES)
    elif model_name in ["UNET", "UNET_AUG"]:
        model = Unet(input_shape=IMG_SHAPE, classes=NUM_CLASSES)
    else:
        raise FileNotFoundError

    model.load_weights(MODEL_WEIGHTS[model_name])
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=metrics)

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
