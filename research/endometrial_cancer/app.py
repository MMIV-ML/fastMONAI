import gradio as gr
import torch
import cv2
from pathlib import Path
from huggingface_hub import snapshot_download
from fastMONAI.vision_all import *

def initialize_system():
    """Initial setup of model paths and other constants."""
    models_path = Path(snapshot_download(repo_id="skaliy/endometrial_cancer_segmentation",
                                         cache_dir='models',
                                         revision='main'))
    save_dir =  Path.cwd() / 'ec_pred'    
    save_dir.mkdir(parents=True, exist_ok=True)
    download_example_endometrial_cancer_data(path=save_dir, multi_channel=False)
    
    return models_path, save_dir


def gradio_image_segmentation(fileobj, learn, reorder, resample, save_dir):
    """Predict function using the learner and other resources."""
    img_path = Path(fileobj.name)

    save_fn = 'pred_' + img_path.stem
    save_path = save_dir / save_fn
    org_img, input_img, org_size = med_img_reader(img_path, 
                                                  reorder=reorder,
                                                  resample=resample,
                                                  only_tensor=False)
    
    mask_data = inference(learn, reorder=reorder, resample=resample,
                          org_img=org_img, input_img=input_img,
                          org_size=org_size).data 
    
    if "".join(org_img.orientation) == "LSA":        
        mask_data = mask_data.permute(0,1,3,2)
        mask_data = torch.flip(mask_data[0], dims=[1])
        mask_data = torch.Tensor(mask_data)[None]

    img = org_img.data
    org_img.set_data(mask_data)
    org_img.save(save_path)

    img, pred_mask = get_mid_slice(img[0], mask_data[0])
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8) #normalize
    
    volume = compute_binary_tumor_volume(org_img)
   
    return get_fused_image(img, pred_mask), round(volume, 2)


models_path, save_dir = initialize_system()
learn, reorder, resample = load_system_resources(models_path=models_path,
                                                 learner_fn='vibe-learner.pkl',
                                                 variables_fn='vars.pkl')
output_text = gr.Textbox(label="Volume of the predicted tumor:")

demo = gr.Interface(
    fn=lambda fileobj: gradio_image_segmentation(fileobj, learn, reorder, resample, save_dir),
    inputs=["file"],
    outputs=["image", output_text],
    examples=[[save_dir/"vibe.nii.gz"]])

demo.launch()