# To run the script in terminal:
# 1. Install fastMONAI version `0.3.2`
# 2. Make the script executable: chmod +x inference_script.py
# 3. Run the script with the following command: python inference_script.py IMG_PATH

import argparse

from fastMONAI.vision_all import *
from huggingface_hub import snapshot_download

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('fn_t2', type=str, help='File name of the t2 input image')
parser.add_argument('fn_vibe', type=str, help='File name of the vibe input image')
parser.add_argument('fn_adc', type=str, help='File name of the adc input image')

args = parser.parse_args()

# Download the model from the study repository and load the exported learner. 
# By default, the latest version from the main branch is downloaded.
models_path = Path(snapshot_download(repo_id="skaliy/endometrial_cancer_segmentation",  cache_dir='models', revision='main'))
learner = load_learner(models_path/'t2-vibe-adc-learner.pkl', cpu=True) #TODO add an option to run on GPU

# Load variables
vars_fn = models_path/'vars.pkl'
_, reorder, resample = load_variables(pkl_fn=vars_fn)

# Set file name from command line argument
img_path = [Path(args.fn_t2), Path(args.fn_vibe), Path(args.fn_adc)]

#Save path
save_path = str(img_path[0]).replace(img_path[0].stem, 'pred_' +img_path[0].stem)

#pred_items
org_img, input_img, org_size = med_img_reader(img_path, reorder=reorder, resample=resample, only_tensor=False)

mask_data = inference(learner, reorder=reorder, resample=resample, org_img=org_img, input_img=input_img, org_size=org_size).data 

if "".join(org_img.orientation) == 'LSA':        
    mask_data = mask_data.permute(0,1,3,2)
    mask_data = torch.flip(mask_data[0], dims=[1])
    mask_data = torch.Tensor(mask_data)[None]
    
org_img.set_data(mask_data)
org_img.save(save_path)