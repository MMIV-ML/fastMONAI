# To run the script in terminal:
# 1. Make the script executable: chmod +x inference_script.py
# 2. Run the script with the following command: python inference_script.py IMG_PATH

import argparse

from fastMONAI.vision_all import *
from huggingface_hub import snapshot_download

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('fn', type=str, help='File name of the input image')
args = parser.parse_args()

# Download the models from the study repository and load exported learners 
models_path = Path(snapshot_download(repo_id="skaliy/spine-segmentation",  cache_dir='models'))
learner_list = list(models_path.glob('*learner.pkl'))
loaded_learners = [load_learner(fn, cpu=True) for fn in learner_list]

# Load variables
vars_fn = models_path/'vars.pkl'
_, reorder, resample = load_variables(pkl_fn=vars_fn)

# Set file name from command line argument
img_fn = args.fn
save_fn = fn.split('.nii')[0] + '_pred.nii.gz'

#pred_items
org_img, input_img, org_size = med_img_reader(img_fn, reorder, resample, only_tensor=False)

#Predict with ensemble
mask_data = [inference(learner, reorder, resample, org_img=org_img, input_img=input_img, org_size=org_size).data for learner in loaded_learners]

# Average the accumulated mask data
mask_data = sum(mask_data)/len(loaded_learners)

# Threshold the averaged mask data to create a binary mask
mask_data = torch.where(mask_data > 0.5, 1., 0.)

# Apply postprocessing to remove small objects from the binary mask
mask_data = refine_binary_pred_mask(mask_data, remove_size=10437, percentage=0.2)

# Set the data of the image object to the processed mask data and save the predicted mask
org_img.set_data(mask_data)
org_img.save(save_fn)