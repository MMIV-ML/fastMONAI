# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_vision_augment.ipynb.

# %% auto 0
__all__ = ['CustomDictTransform', 'do_pad_or_crop', 'PadOrCrop', 'do_resize', 'ZNormalization', 'BraTSMaskConverter',
           'BinaryConverter', 'RandomGhosting', 'RandomSpike', 'RandomNoise', 'RandomBiasField', 'RandomBlur',
           'RandomGamma', 'RandomMotion', 'RandomElasticDeformation', 'RandomAffine', 'RandomFlip', 'OneOf']

# %% ../nbs/03_vision_augment.ipynb 2
from fastai.data.all import *
from .vision_core import *
import torchio as tio

# %% ../nbs/03_vision_augment.ipynb 5
class CustomDictTransform(ItemTransform):
    '''Wrapper to perform an identical transformation on both image and target (if it is a mask) during training.'''
    
    split_idx = 0
    def __init__(self, aug): self.aug = aug

    def encodes(self, x):
        '''Apply transformation to an image, and the same random transformation to the target if it is a mask.

        Args:
            x: Contains image and target.

        Returns:
            MedImage: Transformed image data.
            (MedMask, TensorCategory, ...todo): If the target is a mask, then return a transformed mask data. Otherwise, return target value.
        '''

        img, y_true = x

        if isinstance(y_true, (MedMask)):
            aug = self.aug(tio.Subject(img=tio.ScalarImage(tensor=img, affine=MedImage.affine_matrix), mask=tio.LabelMap(tensor=y_true, affine=MedImage.affine_matrix)))
            return MedImage.create(aug['img'].data), MedMask.create(aug['mask'].data)
        else:
            aug = self.aug(tio.Subject(img=tio.ScalarImage(tensor=img)))
            return MedImage.create(aug['img'].data), y_true

# %% ../nbs/03_vision_augment.ipynb 8
def do_pad_or_crop(o, target_shape, padding_mode, mask_name, dtype=torch.Tensor):

    pad_or_crop = tio.CropOrPad(target_shape=target_shape, padding_mode=padding_mode, mask_name=mask_name)
    return dtype(pad_or_crop(o))

# %% ../nbs/03_vision_augment.ipynb 9
class PadOrCrop(DisplayedTransform):
    '''Resize image using TorchIO `CropOrPad`.'''

    order=0
    def __init__(self, size, padding_mode=0, mask_name=None):
        if not is_listy(size): size=[size,size,size]
        self.size, self.padding_mode, self.mask_name = size, padding_mode, mask_name

    def encodes(self, o:(MedImage, MedMask)):
        return do_pad_or_crop(o,target_shape=self.size, padding_mode=self.padding_mode, mask_name=self.mask_name, dtype=type(o))

# %% ../nbs/03_vision_augment.ipynb 11
def do_resize(o, target_shape, image_interpolation='linear', label_interpolation='nearest'):
    '''Resample images so the output shape matches the given target shape.'''
    resize = tio.Resize(target_shape, image_interpolation=image_interpolation, label_interpolation=label_interpolation)
    return resize(o)

# %% ../nbs/03_vision_augment.ipynb 12
def _do_z_normalization(o, masking_method, channel_wise):

    z_normalization = tio.ZNormalization(masking_method=masking_method)
    normalized_tensor = torch.zeros(o.shape)

    if channel_wise:
        for idx, c in enumerate(o): 
            normalized_tensor[idx] = z_normalization(c[None])[0]
            
    else: normalized_tensor = z_normalization(o)

    return normalized_tensor

# %% ../nbs/03_vision_augment.ipynb 13
class ZNormalization(DisplayedTransform):
    '''Apply TorchIO `ZNormalization`.'''

    order=0
    def __init__(self, masking_method=None, channel_wise=True):
        self.masking_method, self.channel_wise = masking_method, channel_wise

    def encodes(self, o:(MedImage)): return MedImage.create(_do_z_normalization(o, self.masking_method, self.channel_wise))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 15
class BraTSMaskConverter(DisplayedTransform):
    '''Convert BraTS masks.'''

    order=1

    def encodes(self, o:(MedImage)): return o

    def encodes(self, o:(MedMask)):
        o = torch.where(o==4, 3., o)
        return MedMask.create(o)

# %% ../nbs/03_vision_augment.ipynb 17
class BinaryConverter(DisplayedTransform):
    '''Convert to binary mask.'''

    order=1

    def encodes(self, o:(MedImage)): return o

    def encodes(self, o:(MedMask)):
        o = torch.where(o>0, 1., 0)
        return MedMask.create(o)

# %% ../nbs/03_vision_augment.ipynb 19
def _do_rand_ghosting(o, intensity, p):
    
    add_ghosts = tio.RandomGhosting(intensity=intensity, p=p)
    return add_ghosts(o)

# %% ../nbs/03_vision_augment.ipynb 20
class RandomGhosting(DisplayedTransform):
    '''Apply TorchIO `RandomGhosting`.'''

    split_idx,order=0,1

    def __init__(self, intensity =(0.5, 1), p=0.5):
        self.intensity, self.p  = intensity, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_ghosting(o, self.intensity, self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 22
def _do_rand_spike(o, num_spikes, intensity, p):

    add_spikes = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity, p=p)
    return add_spikes(o) #return torch tensor

# %% ../nbs/03_vision_augment.ipynb 23
class RandomSpike(DisplayedTransform):
    '''Apply TorchIO `RandomSpike`.'''
    
    split_idx,order=0,1

    def __init__(self, num_spikes=1, intensity=(1, 3), p=0.5):
        self.num_spikes, self.intensity, self.p  = num_spikes, intensity, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_spike(o, self.num_spikes, self.intensity, self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 25
def _do_rand_noise(o, mean, std, p):

    add_noise = tio.RandomNoise(mean=mean, std=std, p=p)
    return add_noise(o) #return torch tensor

# %% ../nbs/03_vision_augment.ipynb 26
class RandomNoise(DisplayedTransform):
    '''Apply TorchIO `RandomNoise`.'''

    split_idx,order=0,1

    def __init__(self, mean=0, std=(0, 0.25), p=0.5):
        self.mean, self.std, self.p  = mean, std, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_noise(o, mean=self.mean, std=self.std, p=self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 28
def _do_rand_biasfield(o, coefficients, order, p):

    add_biasfield = tio.RandomBiasField(coefficients=coefficients, order=order, p=p)
    return add_biasfield(o) #return torch tensor

# %% ../nbs/03_vision_augment.ipynb 29
class RandomBiasField(DisplayedTransform):
    '''Apply TorchIO `RandomBiasField`.'''

    split_idx,order=0,1

    def __init__(self, coefficients=0.5, order=3, p=0.5):
        self.coefficients, self.order, self.p  = coefficients, order, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_biasfield(o, coefficients=self.coefficients, order=self.order, p=self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 31
def _do_rand_blur(o, std, p):

    add_blur = tio.RandomBlur(std=std, p=p)
    return add_blur(o) 

# %% ../nbs/03_vision_augment.ipynb 32
class RandomBlur(DisplayedTransform):
    '''Apply TorchIO `RandomBiasField`.'''

    split_idx,order=0,1

    def __init__(self, std=(0, 2), p=0.5):
        self.std, self.p  = std, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_blur(o, std=self.std, p=self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 34
def _do_rand_gamma(o, log_gamma, p):

    add_gamma = tio.RandomGamma(log_gamma=log_gamma, p=p)
    return add_gamma(o) 

# %% ../nbs/03_vision_augment.ipynb 35
class RandomGamma(DisplayedTransform):
    '''Apply TorchIO `RandomGamma`.'''


    split_idx,order=0,1

    def __init__(self, log_gamma=(-0.3, 0.3), p=0.5):
        self.log_gamma, self.p  = log_gamma, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_gamma(o, log_gamma=self.log_gamma, p=self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 37
def _do_rand_motion(o, degrees, translation, num_transforms, image_interpolation, p):

    add_motion = tio.RandomMotion(degrees=degrees, translation=translation, num_transforms=num_transforms, image_interpolation=image_interpolation, p=p)
    return add_motion(o) #return torch tensor

# %% ../nbs/03_vision_augment.ipynb 38
class RandomMotion(DisplayedTransform):
    '''Apply TorchIO `RandomMotion`.'''

    split_idx,order=0,1

    def __init__(self, degrees=10, translation=10, num_transforms=2, image_interpolation='linear', p=0.5):
        self.degrees,self.translation, self.num_transforms, self.image_interpolation, self.p = degrees,translation, num_transforms, image_interpolation, p

    def encodes(self, o:(MedImage)): return MedImage.create(_do_rand_motion(o, degrees=self.degrees,translation=self.translation, num_transforms=self.num_transforms, image_interpolation=self.image_interpolation, p=self.p))
    def encodes(self, o:(MedMask)):return o

# %% ../nbs/03_vision_augment.ipynb 41
class RandomElasticDeformation(CustomDictTransform):
    '''Apply TorchIO `RandomElasticDeformation`.'''

    def __init__(self,num_control_points=7, max_displacement=7.5, image_interpolation='linear', p=0.5): 
        super().__init__(tio.RandomElasticDeformation(num_control_points=num_control_points, max_displacement=max_displacement, image_interpolation=image_interpolation, p=p))

# %% ../nbs/03_vision_augment.ipynb 43
class RandomAffine(CustomDictTransform):
    '''Apply TorchIO `RandomAffine`.'''

    def __init__(self, scales=0, degrees=10, translation=0, isotropic=False, image_interpolation='linear', default_pad_value=0., p=0.5): 
        super().__init__(tio.RandomAffine(scales=scales, degrees=degrees, translation=translation, isotropic=isotropic, image_interpolation=image_interpolation, default_pad_value=default_pad_value, p=p))

# %% ../nbs/03_vision_augment.ipynb 45
class RandomFlip(CustomDictTransform):
    '''Apply TorchIO `RandomFlip`.'''

    def __init__(self, axes='LR', p=0.5):
        super().__init__(tio.RandomFlip(axes=axes, flip_probability=p))

# %% ../nbs/03_vision_augment.ipynb 47
class OneOf(CustomDictTransform):
    '''Apply only one of the given transforms using TorchIO `OneOf`.'''

    def __init__(self, transform_dict, p=1):
        super().__init__(tio.OneOf(transform_dict, p=p))
