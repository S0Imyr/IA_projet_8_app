from django import forms


class ImageMaskForm(forms.Form):
    image = forms.ImageField()
    mask = forms.ImageField(required=False)
