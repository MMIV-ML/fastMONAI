# To run the script in terminal:
# 1. Make the script executable: chmod +x inference_script.py
# 2. Run the script with the following command: python inference_script.py IMG_PATH

import argparse

from fastMONAI.vision_all import *
from IPython.display import clear_output

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('fn', type=str, help='File name of the input image')
args = parser.parse_args()

# Load variables
vars_fn = glob.glob('models/*/snapshots/*/vars.pkl')[0]
_, reorder, resample = load_variables(pkl_fn=vars_fn)

# Load models
models = [load_learner(f'models/{i}_learner.pkl', cpu=False) for i in range(3)]

# Set file name from command line argument
fn = args.fn
save_fn = fn.split('.nii')[0] + '_pred.nii.gz'

# Initialize mask data
mask_data = None

# Infer mask for each model
for learner in models:
    mask = inference(learner, reorder, resample, fn)
    
    # Initialize mask data if necessary
    if mask_data is None:
        mask_data = torch.zeros_like(mask.data)
    # Add mask data to accumulated mask data
    else:
        mask_data += mask.data

# Average the accumulated mask data
mask_data /= len(models)

# Threshold the averaged mask data to create a binary mask
mask_data = torch.where(mask_data > 0.5, 1., 0.)

# Apply postprocessing to remove small objects from the binary mask
mask_data = torch.Tensor(pred_postprocess(mask_data))

# Set the data of the mask object to the processed mask data
mask.set_data(mask_data)

# Save the mask
mask.save(save_fn)