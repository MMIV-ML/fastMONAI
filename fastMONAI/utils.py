# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_utils.ipynb.

# %% auto 0
__all__ = ['store_variables', 'load_variables', 'print_colab_gpu_info']

# %% ../nbs/06_utils.ipynb 1
import pickle
import torch

# %% ../nbs/06_utils.ipynb 3
def store_variables(pkl_fn:str, # Filename of the pickle file
                    var_vals:list # A list of variable values
                   ) -> None:
    '''Save variable values in a pickle file.'''

    with open(pkl_fn, 'wb') as f:
        pickle.dump(var_vals, f)

# %% ../nbs/06_utils.ipynb 4
def load_variables(pkl_fn # Filename of the pickle file
                  ):
    '''Load stored variable values from a pickle file.

    Returns: A list of variable values.
    '''

    with open(pkl_fn, 'rb') as f:
        return pickle.load(f)

# %% ../nbs/06_utils.ipynb 5
def print_colab_gpu_info(): 
    '''Check if we have a GPU attached to the runtime.'''
    
    colab_gpu_msg =(f"{'#'*80}\n"
                    "Remember to attach a GPU to your Colab Runtime:"
                    "\n1. From the **Runtime** menu select **Change Runtime Type**"
                    "\n2. Choose **GPU** from the drop-down menu"
                    "\n3. Click **'SAVE'**\n"
                    f"{'#'*80}")
    
    if torch.cuda.is_available(): print('GPU attached.')
    else: print(colab_gpu_msg)
