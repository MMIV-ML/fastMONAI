# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/09_external_data.ipynb.

# %% auto 0
__all__ = ['MURLs', 'download_ixi_data', 'download_ixi_tiny', 'download_spine_test_data', 'download_example_spine_data',
           'download_and_process_MedMNIST3D']

# %% ../nbs/09_external_data.ipynb 1
from pathlib import Path
from glob import glob
from numpy import load 
import pandas as pd
from monai.apps import download_url, download_and_extract
from torchio.datasets.ixi import IXITiny
from torchio import ScalarImage
import multiprocessing as mp
from functools import partial

# %% ../nbs/09_external_data.ipynb 3
class MURLs():
    """A class with external medical dataset URLs."""

    IXI_DATA = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'
    IXI_DEMOGRAPHIC_INFORMATION = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'
    CHENGWEN_CHU_SPINE_DATA = 'https://drive.google.com/uc?id=1rbm9-KKAexpNm2mC9FsSbfnS8VJaF3Kn&confirm=t'
    EXAMPLE_SPINE_DATA = 'https://drive.google.com/uc?id=1Ms3Q6MYQrQUA_PKZbJ2t2NeYFQ5jloMh'
    #NODULE_MNIST_DATA = 'https://zenodo.org/record/6496656/files/nodulemnist3d.npz?download=1'
    MEDMNIST_DICT = {'OrganMNIST3D': 'https://zenodo.org/record/6496656/files/organmnist3d.npz?download=1',	
                     'NoduleMNIST3D': 'https://zenodo.org/record/6496656/files/nodulemnist3d.npz?download=1',
                     'AdrenalMNIST3D': 'https://zenodo.org/record/6496656/files/adrenalmnist3d.npz?download=1',	
                     'FractureMNIST3D': 'https://zenodo.org/record/6496656/files/fracturemnist3d.npz?download=1',
                     'VesselMNIST3D': 'https://zenodo.org/record/6496656/files/vesselmnist3d.npz?download=1', 
                     'SynapseMNIST3D': 'https://zenodo.org/record/6496656/files/synapsemnist3d.npz?download=1'}

# %% ../nbs/09_external_data.ipynb 4
def _process_ixi_xls(xls_path: (str, Path), img_path: Path) -> pd.DataFrame:
    """Private method to process the demographic information for the IXI dataset.

    Args:
        xls_path: File path to the xls file with the demographic information.
        img_path: Folder path to the images.

    Returns:
        A processed dataframe with image path and demographic information.

    Raises:
        ValueError: If xls_path or img_path do not exist.
    """

    print('Preprocessing ' + str(xls_path))

    df = pd.read_excel(xls_path)

    duplicate_subject_ids = df[df.duplicated(['IXI_ID'], keep=False)].IXI_ID.unique()

    for subject_id in duplicate_subject_ids:
        age = df.loc[df.IXI_ID == subject_id].AGE.nunique()
        if age != 1: df = df.loc[df.IXI_ID != subject_id]  # Remove duplicates with two different age values

    df = df.drop_duplicates(subset='IXI_ID', keep='first').reset_index(drop=True)

    df['subject_id'] = ['IXI' + str(subject_id).zfill(3) for subject_id in df.IXI_ID.values]
    df = df.rename(columns={'SEX_ID (1=m, 2=f)': 'gender'})
    df['age_at_scan'] = df.AGE.round(2)
    df = df.replace({'gender': {1: 'M', 2: 'F'}})

    img_list = list(img_path.glob('*.nii.gz'))
    for path in img_list:
        subject_id = path.parts[-1].split('-')[0]
        df.loc[df.subject_id == subject_id, 't1_path'] = str(path)

    df = df.dropna()
    df = df[['t1_path', 'subject_id', 'gender', 'age_at_scan']]
    
    return df

# %% ../nbs/09_external_data.ipynb 6
def download_ixi_data(path: (str, Path) = '../data') -> Path:
    """Download T1 scans and demographic information from the IXI dataset.
    
    Args:
        path: Path to the directory where the data will be stored. Defaults to '../data'.

    Returns:
        The path to the stored CSV file.
    """

    path = Path(path) / 'IXI'
    img_path = path / 'T1_images'

    # Check whether image data already present in img_path:
    is_extracted = False
    try:
        if len(list(img_path.iterdir())) >= 581:  # 581 imgs in the IXI dataset
            is_extracted = True
            print(f"Images already downloaded and extracted to {img_path}")
    except:
        is_extracted = False

    if not is_extracted:
        download_and_extract(url=MURLs.IXI_DATA, filepath=path / 'IXI-T1.tar', output_dir=img_path)
        (path / 'IXI-T1.tar').unlink()

    download_url(url=MURLs.IXI_DEMOGRAPHIC_INFORMATION, filepath=path / 'IXI.xls')

    processed_df = _process_ixi_xls(xls_path=path / 'IXI.xls', img_path=img_path)
    processed_df.to_csv(path / 'dataset.csv', index=False)

    return path

# %% ../nbs/09_external_data.ipynb 8
def download_ixi_tiny(path: (str, Path) = '../data') -> Path:
    """Download the tiny version of the IXI dataset provided by TorchIO.

    Args:
        path: The directory where the data will be 
            stored. If not provided, defaults to '../data'.

    Returns:
        The path to the directory where the data is stored.
    """
    
    path = Path(path) / 'IXITiny'
    
    IXITiny(root=str(path), download=True)
    download_url(url=MURLs.IXI_DEMOGRAPHIC_INFORMATION, filepath=path/'IXI.xls')
    
    processed_df = _process_ixi_xls(xls_path=path/'IXI.xls', img_path=path/'image')
    processed_df['labels'] = processed_df['t1_path'].str.replace('image','label')
    
    processed_df.to_csv(path/'dataset.csv', index=False)
    
    return path

# %% ../nbs/09_external_data.ipynb 10
def _create_spine_df(dir: Path) -> pd.DataFrame:
    """Create a pandas DataFrame containing information about spinal images.

    Args:
        dir: Directory path where data (image and segmentation 
            mask files) are stored.

    Returns:
         A DataFrame containing the paths to the image files and their 
            corresponding mask files, the subject IDs, and a flag indicating that 
            these are test data.
    """
    
    img_list = glob(str(dir / 'img/*.nii.gz'))
    mask_list = [str(fn).replace('img', 'seg') for fn in img_list]
    subject_id_list = [fn.split('_')[-1].split('.')[0] for fn in mask_list]
    
    test_data = {
        't2_img_path': img_list,
        't2_mask_path': mask_list,
        'subject_id': subject_id_list,
        'is_test': True,
    }

    return pd.DataFrame(test_data)

# %% ../nbs/09_external_data.ipynb 11
def download_spine_test_data(path: (str, Path) = '../data') -> pd.DataFrame:
    """Downloads T2w scans from the study 'Fully Automatic Localization and 
    Segmentation of 3D Vertebral Bodies from CT/MR Images via a Learning-Based 
    Method' by Chu et. al. 

    Args:
        path: Directory where the downloaded data 
            will be stored and extracted. Defaults to '../data'.

    Returns:
        Processed dataframe containing image paths, label paths, and subject IDs.
    """
    
    study = 'chengwen_chu_2015'
    
    download_and_extract(
        url=MURLs.CHENGWEN_CHU_SPINE_DATA, 
        filepath=f'{study}.zip', 
        output_dir=path
    )
    Path(f'{study}.zip').unlink()
    
    return _create_spine_df(Path(path) / study)

# %% ../nbs/09_external_data.ipynb 12
def download_example_spine_data(path: (str, Path) = '../data') -> Path:
    """Downloads example T2w scan and corresponding predicted mask.
    
    Args:
        path: Directory where the downloaded data 
            will be stored and extracted. Defaults to '../data'.

    Returns:
        Path to the directory where the example data has been extracted.
    """
    
    study = 'example_data'
    
    download_and_extract(
        url=MURLs.EXAMPLE_SPINE_DATA, 
        filepath='example_data.zip', 
        output_dir=path
    )
    Path('example_data.zip').unlink()
    
    return Path(path) / study

# %% ../nbs/09_external_data.ipynb 18
def _process_medmnist_img(path, idx_arr):
    """Save tensor as NIfTI."""
    
    idx, arr = idx_arr
    img = ScalarImage(tensor=arr[None, :])
    fn = path/f'{idx}_nodule.nii.gz'
    img.save(fn)
    return str(fn)

# %% ../nbs/09_external_data.ipynb 19
def _df_sort_and_add_columns(df, label_list, is_val):
    """Sort the dataframe based on img_idx and add labels and if it is validation data column."""
    
    df = df.sort_values(by='img_idx').reset_index(drop=True)
    df['labels'], df['is_val'] = label_list, is_val     
    #df = df.replace({"labels": {0:'b', 1:'m'}})
    df = df.drop('img_idx', axis=1)
    
    return df 

# %% ../nbs/09_external_data.ipynb 20
def _create_nodule_df(pool, output_dir, imgs, labels, is_val=False): 
    """Create dataframe for MedMNIST data."""
    
    img_path_list = pool.map(partial(_process_medmnist_img, output_dir), enumerate(imgs))
    img_idx = [float(Path(fn).parts[-1].split('_')[0]) for fn in img_path_list]
    
    df = pd.DataFrame(list(zip(img_path_list, img_idx)), columns=['img_path','img_idx'])        
    return  _df_sort_and_add_columns(df, labels, is_val)

# %% ../nbs/09_external_data.ipynb 21
def download_and_process_MedMNIST3D(study: str, 
                                    path: (str, Path) = '../data', 
                                    max_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads and processes a particular MedMNIST dataset.

    Args:
        study: select MedMNIST dataset ('OrganMNIST3D', 'NoduleMNIST3D', 
                'AdrenalMNIST3D', 'FractureMNIST3D', 'VesselMNIST3D', 'SynapseMNIST3D')
        path: Directory where the downloaded data
            will be stored and extracted. Defaults to '../data'.
        max_workers: Maximum number of worker processes to use
            for data processing. Defaults to 1.

    Returns:
        Two pandas DataFrames. The first DataFrame combines training and validation data, 
        and the second DataFrame contains the testing data.
    """
    path = Path(path) / study
    dataset_file_path = path / f'{study}.npz'

    try: 
        download_url(url=MURLs.MEDMNIST_DICT[study], filepath=dataset_file_path)
    except: 
        raise ValueError(f"Dataset '{study}' does not exist.")

    data = load(dataset_file_path)
    keys = ['train_images', 'val_images', 'test_images']

    for key in keys: 
        (path / key).mkdir(exist_ok=True)
    
    train_imgs, val_imgs, test_imgs = data[keys[0]], data[keys[1]], data[keys[2]]

    # Process the data and create DataFrames
    with mp.Pool(processes=max_workers) as pool:
        train_df = _create_nodule_df(pool, path / keys[0], train_imgs, data['train_labels'])
        val_df = _create_nodule_df(pool, path / keys[1], val_imgs, data['val_labels'], is_val=True)
        test_df = _create_nodule_df(pool, path / keys[2], test_imgs, data['test_labels'])

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    dataset_file_path.unlink()

    return train_val_df, test_df

