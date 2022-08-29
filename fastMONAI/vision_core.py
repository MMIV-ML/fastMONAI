# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_vision_core.ipynb.

# %% auto 0
__all__ = ['med_img_reader', 'MetaResolver', 'MedBase', 'MedImage', 'MedMask']

# %% ../nbs/01_vision_core.ipynb 2
from .vision_plot import *
from fastai.data.all import *
from torchio import ScalarImage, LabelMap, ToCanonical, Resample
import pickle
import warnings

# %% ../nbs/01_vision_core.ipynb 5
def _load(fn:str, dtype=None):
    '''Private method to load image as either ScalarImage or LabelMap.

    Args:
        fn : Image path.
        dtype: Datatype.

    Returns:
        (ScalarImage, LabelMap): An object that contains a 4D tensor and metadata.
    '''

    if dtype is MedMask: return LabelMap(fn)
    else: return ScalarImage(fn)

# %% ../nbs/01_vision_core.ipynb 6
def _multi_channel(img_fns:list, dtype=None):
    '''Private method to load multisequence data.

    Args:
        img_fns: List of image paths s(e.g. T1, T2, T1CE, DWI).

    Returns:
        torch.Tensor: A stacked 4D tensor.
    '''

    img_list = []
    for fn in img_fns:
        o = _load(o, dtype=dtype).data[0]
        img_list.append(o)

    return torch.stack(img_list, dim=0)

# %% ../nbs/01_vision_core.ipynb 7
def med_img_reader(fn:(str, Path), # Image path
                   dtype=torch.Tensor, # Datatype (MedImage, MedMask, torch.Tensor)
                   resample:list=None, # Wheter to resample image to different voxel sizes and image dimensions.
                   reorder:bool=False, # Wheter to reorder the data to be closest to canonical (RAS+) orientation.
                   only_tensor:bool=True # Wheter to return only image tensor. If False, return ScalerImage or LabelMap object.
                  ):
    '''Load a medical image data. 4D tensor is returned as `dtype` if `only_tensor` is True, otherwise return ScalerImage or LabelMap.'''

    if ';' in fn:
        img_fns = o.split(';')
        return _multi_channel(img_fns, dtype=dtype)

    o = _load(fn, dtype=dtype)
    org_metadata = {}
    
    if reorder:
        org_metadata['orientation'] = o.orientation
        transform = ToCanonical()
        o = transform(o)
    
    if resample and not all(np.isclose(o.spacing, resample)):
            org_metadata['spacing'], org_metadata['org_size']  = list(o.spacing), o.shape[1:]

            transform = Resample(resample)
            o = transform(o)

    if only_tensor: return dtype(o.data.type(torch.float))

    return o, org_metadata

# %% ../nbs/01_vision_core.ipynb 9
class MetaResolver(type(torch.Tensor), metaclass=BypassNewMeta):
    '''A class to bypass metaclass conflict:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/batch.html
    '''

    pass

# %% ../nbs/01_vision_core.ipynb 10
class MedBase(torch.Tensor, metaclass=MetaResolver):
    '''A class that represents an image object. Metaclass casts x to this class if it is of type cls._bypass_type.'''

    _bypass_type=torch.Tensor
    _show_args = {'cmap':'gray'}
    resample, reorder = None, False

    @classmethod
    def create(cls, fn:(Path,str, torch.Tensor), **kwargs):
        '''Open an medical image and cast to MedBase object. If it is a torch.Tensor cast to MedBase object.

        Args:
            fn: Image path or a 4D torch.Tensor.
            kwargs: additional parameters.

        Returns:
            A 4D tensor as MedBase object.
        '''

        if isinstance(fn, torch.Tensor): return cls(fn)
        return med_img_reader(fn, dtype=cls, resample=cls.resample, reorder=cls.reorder)

    @classmethod
    def item_preprocessing(cls, resample:(list, int, tuple), reorder:bool):
        '''Change the values for the class variables `resample` and `reorder`.

        Args:
            resample: A list with voxel spacing.
            reorder: Wheter to reorder the data to be closest to canonical (RAS+) orientation.
        '''

        cls.resample = resample
        cls.reorder = reorder

    def show(self, ctx=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
        "Show Medimage using `merge(self._show_args, kwargs)`"
        return show_med_img(self, ctx=ctx, channel=channel, indices=indices, anatomical_plane=anatomical_plane, voxel_size=self.resample,  **merge(self._show_args, kwargs))

    def __repr__(self): return f'{self.__class__.__name__} mode={self.mode} size={"x".join([str(d) for d in self.size])}'

# %% ../nbs/01_vision_core.ipynb 11
class MedImage(MedBase):
    '''Subclass of MedBase that represents an image object.'''
    pass

# %% ../nbs/01_vision_core.ipynb 12
class MedMask(MedBase):
    '''Subclass of MedBase that represents an mask object.'''
    _show_args = {'alpha':0.5, 'cmap':'tab20'}