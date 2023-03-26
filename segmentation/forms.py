from django import forms

MODEL_CHOICES = (('FPN', 'FPN sans augmentation'),
                 ('FPN_AUG', 'FPN avec augmentation'),
                 ('UNET', 'U-Net sans augmentation'),
                 ('UNET_AUG', 'U-Net avec augmentation'),
                )


class ImageMaskForm(forms.Form):
    image = forms.ImageField()
    mask = forms.ImageField(required=False)
    model = forms.ChoiceField(choices=MODEL_CHOICES, widget=forms.RadioSelect)
