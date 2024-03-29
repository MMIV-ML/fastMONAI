# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_vision_inference.ipynb.

# %% auto 0
__all__ = ['save_series_pred', 'load_system_resources', 'inference', 'compute_binary_tumor_volume', 'refine_binary_pred_mask',
           'gradio_image_classifier']

# %% ../nbs/06_vision_inference.ipynb 1
from copy import copy
from pathlib import Path
import torch
import numpy as np
from scipy.ndimage import label
from skimage.morphology import remove_small_objects
from SimpleITK import DICOMOrient, GetArrayFromImage
from torchio import Resize, Image
from .vision_core import *
from .vision_augmentation import do_pad_or_crop
from .utils import load_variables
from imagedata.series import Series
from fastai.learner import load_learner

# %% ../nbs/06_vision_inference.ipynb 3
def _update_uid(attribute, series_obj, val='1234', slice_idx=None):
    """Updates a DICOM UID by replacing its last 4 characters with the provided value."""
    
    uid = series_obj.getDicomAttribute(attribute, slice=slice_idx)[:-4] + val
    series_obj.setDicomAttribute(attribute, uid, slice=slice_idx)
    return series_obj

# %% ../nbs/06_vision_inference.ipynb 4
def save_series_pred(series_obj, save_dir, val='1234'):
    """Saves series prediction with updated DICOM UIDs."""
    
    series_obj.seriesInstanceUID = series_obj.seriesInstanceUID[:-4] + val
    
    for slice_idx in range(series_obj.slices):
        series_obj = _update_uid('SOPInstanceUID', series_obj, val, slice_idx)
        series_obj = _update_uid('SeriesInstanceUID', series_obj, val, slice_idx)
        
    series_obj.write(save_dir, opts={'keep_uid': True}, formats=['dicom'])

# %% ../nbs/06_vision_inference.ipynb 5
def _to_original_orientation(input_img, org_orientation):
    """Reorients the image to its original orientation."""
    
    orientation_itk = DICOMOrient(input_img, org_orientation)
    reoriented_array =  GetArrayFromImage(orientation_itk).transpose()
    
    return reoriented_array[None]

# %% ../nbs/06_vision_inference.ipynb 6
def _do_resize(o, target_shape, image_interpolation='linear', 
               label_interpolation='nearest'):
    """Resample images so the output shape matches the given target shape."""

    resize = Resize(
        target_shape, 
        image_interpolation=image_interpolation, 
        label_interpolation=label_interpolation
    )
    
    return resize(o)

# %% ../nbs/06_vision_inference.ipynb 7
def load_system_resources(models_path, learner_fn, variables_fn):
    """Load necessary resources like learner and variables."""

    learn = load_learner(models_path / learner_fn, cpu=True) 
    vars_fn = models_path / variables_fn
    _, reorder, resample = load_variables(pkl_fn=vars_fn)

    return learn, reorder, resample

# %% ../nbs/06_vision_inference.ipynb 8
def inference(learn_inf, reorder, resample, fn: (str, Path) = '',
              save_path: (str, Path) = None, org_img=None, input_img=None,
              org_size=None): 
    """Predict on new data using exported model."""         
    
    if None in [org_img, input_img, org_size]: 
        org_img, input_img, org_size = med_img_reader(fn, reorder, resample, 
                                                      only_tensor=False)
    else: 
        org_img, input_img = copy(org_img), copy(input_img)
    
    pred, *_ = learn_inf.predict(input_img.data)
    
    pred_mask = do_pad_or_crop(pred.float(), input_img.shape[1:], padding_mode=0, 
                               mask_name=None)
    input_img.set_data(pred_mask)
    
    input_img = _do_resize(input_img, org_size, image_interpolation='nearest')
    
    reoriented_array = _to_original_orientation(input_img.as_sitk(), 
                                                ('').join(org_img.orientation))
    
    org_img.set_data(reoriented_array)

    if save_path:
        save_fn = Path(save_path)/('pred_' + Path(fn).parts[-1])
        org_img.save(save_fn)
        return save_fn
    
    return org_img

# %% ../nbs/06_vision_inference.ipynb 9
def compute_binary_tumor_volume(mask_data: Image):
    """Compute the volume of the tumor in milliliters (ml)."""
    
    dx, dy, dz = mask_data.spacing
    voxel_volume_ml = dx * dy * dz / 1000  
    return np.sum(mask_data) * voxel_volume_ml

# %% ../nbs/06_vision_inference.ipynb 11
def refine_binary_pred_mask(pred_mask, 
                            remove_size: (int, float) = None,
                            percentage: float = 0.2,
                            verbose: bool = False) -> torch.Tensor:
    """Removes small objects from the predicted binary mask.

    Args:
        pred_mask: The predicted mask from which small objects are to be removed.
        remove_size: The size under which objects are considered 'small'.
        percentage: The percentage of the remove_size to be used as threshold. 
            Defaults to 0.2.
        verbose: If True, print the number of components. Defaults to False.

    Returns:
        The processed mask with small objects removed.
    """
                                
    labeled_mask, n_components = label(pred_mask)

    if verbose:
        print(n_components)

    if remove_size is None:
        sizes = np.bincount(labeled_mask.ravel())
        max_label = sizes[1:].argmax() + 1
        remove_size = sizes[max_label]

    small_objects_threshold = remove_size * percentage
    processed_mask = remove_small_objects(
        labeled_mask, min_size=small_objects_threshold)

    return torch.Tensor(processed_mask > 0).float()                          

# %% ../nbs/06_vision_inference.ipynb 13
def gradio_image_classifier(file_obj, learn, reorder, resample):
    """Predict on images using exported learner and return the result as a dictionary."""
    
    img_path = Path(file_obj.name)
    img = med_img_reader(img_path, reorder=reorder, resample=resample)
    
    _, _, predictions = learn.predict(img)
    prediction_dict = {index: value.item() for index, value in enumerate(predictions)}

    return prediction_dict
