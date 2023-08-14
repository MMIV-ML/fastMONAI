import gradio as gr
from fastMONAI.vision_all import *
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import cv2

def initialize_system():
    """Initial setup of model paths and other constants."""
    models_path = Path(snapshot_download(repo_id="skaliy/endometrial_cancer_segmentation", cache_dir='models', revision='main'))
    save_dir =  Path.cwd() / 'ec_pred'    
    save_dir.mkdir(parents=True, exist_ok=True)
    download_example_endometrial_cancer_data(path=save_dir, multi_channel=False)
    
    return models_path, save_dir

def load_system_resources(models_path):
    """Load necessary resources like learner and variables."""
    
    learner = load_learner(models_path / 'vibe-learner.pkl', cpu=True)  # TODO: add an option to run on GPU
    vars_fn = models_path / 'vars.pkl'
    _, reorder, resample = load_variables(pkl_fn=vars_fn)

    return learner, reorder, resample

def get_mid_slice(img, mask_data): 
    """Extract the middle slice of the mask in a 3D array."""
    
    sums = mask_data.sum(axis=(0,1))
    mid_idx = np.argmax(sums)
    img, mask_data = img[:, :, mid_idx], mask_data[:, :, mid_idx]
    
    return np.fliplr(np.rot90(img, -1)), np.fliplr(np.rot90(mask_data, -1))


def get_fused_image(img, pred_mask, alpha=0.8):
    """Overlay the mask on the image."""
    gray_img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_color = np.array([0, 0, 255])
    colored_mask = (pred_mask[..., None] * mask_color).astype(np.uint8)
    
    return cv2.addWeighted(gray_img_colored, alpha, colored_mask, 1 - alpha, 0)

def compute_tumor_volume(mask_data):
    """Compute the volume of the tumor in milliliters (ml)."""
    
    dx, dy, dz = mask_data.spacing
    voxel_volume_ml = dx * dy * dz / 1000  
    return np.sum(mask_data) * voxel_volume_ml

def predict(fileobj, learner, reorder, resample, save_dir):
    """Predict function using the learner and other resources."""
    img_path = Path(fileobj.name)

    save_fn = 'pred_' + img_path.stem
    save_path = save_dir / save_fn
    org_img, input_img, org_size = med_img_reader(img_path, reorder=reorder, resample=resample, only_tensor=False)
    
    mask_data = inference(learner, reorder=reorder, resample=resample, org_img=org_img, input_img=input_img, org_size=org_size).data 
    
    if "".join(org_img.orientation) == "LSA":        
        mask_data = mask_data.permute(0,1,3,2)
        mask_data = torch.flip(mask_data[0], dims=[1])
        mask_data = torch.Tensor(mask_data)[None]

    img = org_img.data #TEMP
    
    org_img.set_data(mask_data)
    org_img.save(save_path)

    img, pred_mask = get_mid_slice(img[0], mask_data[0])
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8) #normalize
    volume = compute_tumor_volume(org_img)
   
    return get_fused_image(img, pred_mask), round(volume, 2)


models_path, save_dir = initialize_system()
learner, reorder, resample = load_system_resources(models_path)

#output_image = gr.outputs.Image(label="Segmentation Image")
output_text = gr.Textbox(label="Volume of the predicted tumor:")

demo = gr.Interface(
    fn=lambda fileobj: predict(fileobj, learner, reorder, resample, save_dir),
    inputs=["file"],
    outputs=["image", output_text],
    examples=[[save_dir/"vibe.nii.gz"]]
)

demo.launch()