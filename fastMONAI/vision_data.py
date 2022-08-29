# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_vision_data.ipynb.

# %% auto 0
__all__ = ['pred_to_multiclass_mask', 'batch_pred_to_multiclass_mask', 'pred_to_binary_mask', 'MedDataBlock', 'MedMaskBlock',
           'MedImageDataLoaders', 'show_batch', 'show_results', 'plot_top_losses']

# %% ../nbs/02_vision_data.ipynb 2
from fastai.data.all import *
from fastai.vision.data import *
from .vision_core import *

# %% ../nbs/02_vision_data.ipynb 5
def pred_to_multiclass_mask(pred:torch.Tensor # [C,W,H,D] activation tensor
                           ) -> torch.Tensor:
    '''Apply Softmax function on the predicted tensor to rescale the values in the range [0, 1] and sum to 1.
    Then apply argmax to get the indices of the maximum value of all elements in the predicted Tensor.
    Returns: Predicted mask.
    '''
    pred = pred.softmax(dim=0)
    return pred.argmax(dim=0, keepdims=True)

# %% ../nbs/02_vision_data.ipynb 6
def batch_pred_to_multiclass_mask(pred:torch.Tensor # [B, C, W, H, D] batch of activations
                                 ) -> (torch.Tensor, int):
    '''Convert a batch of predicted activation tensors to masks.
    Returns batch of predicted masks and number of classes.
    '''

    n_classes = pred.shape[1]
    pred = [pred_to_multiclass_mask(p) for p in pred]

    return torch.stack(pred), n_classes

# %% ../nbs/02_vision_data.ipynb 7
def pred_to_binary_mask(pred # [B, C, W, H, D] or [C, W, H, D] activation tensor
                       ) -> torch.Tensor:
    '''Apply Sigmoid function that squishes activations into a range between 0 and 1.
    Then we classify all values greater than or equal to 0.5 to 1, and the values below 0.5 to 0.

    Returns predicted binary mask(s).
    '''

    pred = torch.sigmoid(pred)
    return torch.where(pred>=0.5, 1, 0)

# %% ../nbs/02_vision_data.ipynb 9
class MedDataBlock(DataBlock):
    '''Container to quickly build dataloaders.'''

    def __init__(self, blocks:list=None,dl_type:TfmdDL=None, getters:list=None, n_inp:int=None, item_tfms:list=None,
                 batch_tfms:list=None, reorder:bool=False, resample:(int, list)=None, **kwargs):

        super().__init__(blocks, dl_type, getters, n_inp, item_tfms, batch_tfms, **kwargs)
        MedBase.item_preprocessing(resample,reorder)

# %% ../nbs/02_vision_data.ipynb 12
def MedMaskBlock():
    return TransformBlock(type_tfms=MedMask.create)

# %% ../nbs/02_vision_data.ipynb 14
class MedImageDataLoaders(DataLoaders):
    '''Higher-level `MedDataBlock` API.'''

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, reorder=False, resample=None, **kwargs):
        '''Create from DataFrame.'''

        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)


        dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), y_block), get_x=ColReader(fn_col, suff=suff),
                              get_y=ColReader(label_col, label_delim=label_delim),
                              splitter=splitter,
                              item_tfms=item_tfms,
                              reorder=reorder,
                              resample=resample)

        return cls.from_dblock(dblock, df, **kwargs)

# %% ../nbs/02_vision_data.ipynb 19
@typedispatch
def show_batch(x:MedImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    '''Showing a batch of samples for classification and regression tasks.'''

    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    n = 1 if y is None else 2
    for i in range(n):
        ctxs = [b.show(ctx=c, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs,range(max_n))]

    plt.tight_layout()
    return ctxs

# %% ../nbs/02_vision_data.ipynb 20
@typedispatch
def show_batch(x:MedImage, y:MedMask, samples, ctxs=None, max_n=6, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    '''Showing a batch of decoded segmentation samples.'''

    nrows, ncols = min(len(samples), max_n), x.shape[1] + 1
    imgs = []

    fig,axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = axs.flatten()

    for img, mask in list(zip(x,y)):
        im_channels = [MedImage(c_img[None]) for c_img in img]
        im_channels.append(MedMask(mask))
        imgs.extend(im_channels)

    ctxs = [im.show(ax=ax, indices=indices, anatomical_plane=anatomical_plane) for im, ax in zip(imgs, axs)]
    plt.tight_layout()

    return ctxs

# %% ../nbs/02_vision_data.ipynb 22
@typedispatch
def show_results(x:MedImage, y:torch.Tensor, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    '''Showing samples and their corresponding predictions for regression tasks.'''

    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)

    for i in range(len(samples[0])):
        ctxs = [b.show(ctx=c, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs,range(max_n))]
    for i in range(len(outs[0])):
        ctxs = [b.show(ctx=c, **kwargs) for b,c,_ in zip(outs.itemgot(i),ctxs,range(max_n))]
    return ctxs

# %% ../nbs/02_vision_data.ipynb 23
@typedispatch
def show_results(x:MedImage, y:TensorCategory, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    '''Showing samples and their corresponding predictions for classification tasks.'''

    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(2):
        ctxs = [b.show(ctx=c, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs,range(max_n))]
    ctxs = [r.show(ctx=c, color='green' if b==r else 'red', **kwargs) for b,r,c,_ in zip(samples.itemgot(1),outs.itemgot(0),ctxs,range(max_n))]
    return ctxs

# %% ../nbs/02_vision_data.ipynb 24
@typedispatch
def show_results(x:MedImage, y:MedMask, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=1, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    ''' Showing decoded samples and their corresponding predictions for segmentation tasks.'''

    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize, double=True, title='Target/Prediction')
    for i in range(2):
        ctxs[::2] = [b.show(ctx=c, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs[::2],range(2*max_n))]
    for o in [samples,outs]:
        ctxs[1::2] = [b.show(ctx=c, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,c,_ in zip(o.itemgot(0),ctxs[1::2],range(2*max_n))]
    return ctxs

# %% ../nbs/02_vision_data.ipynb 26
@typedispatch
def plot_top_losses(x: MedImage, y, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    '''Show images in top_losses along with their prediction, actual, loss, and probability of actual class.'''

    title = 'Prediction/Actual/Loss' if type(y) == torch.Tensor else 'Prediction/Actual/Loss/Probability'
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, figsize=figsize, title=title)
    for ax,s,o,r,l in zip(axs, samples, outs, raws, losses):
        s[0].show(ctx=ax, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs)
        if type(y) == torch.Tensor: ax.set_title(f'{r.max().item():.2f}/{s[1]} / {l.item():.2f}')
        else: ax.set_title(f'{o[0]}/{s[1]} / {l.item():.2f} / {r.max().item():.2f}')

# %% ../nbs/02_vision_data.ipynb 27
@typedispatch
def plot_top_losses(x: MedImage, y:TensorMultiCategory, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, channel=0, indices=None, anatomical_plane=0, **kwargs):
    #TODO: not tested yet
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, figsize=figsize)
    for i,(ax,s) in enumerate(zip(axs, samples)): s[0].show(ctx=ax, title=f'Image {i}', channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs)
    rows = get_empty_df(len(samples))
    outs = L(s[1:] + o + (TitledStr(r), TitledFloat(l.item())) for s,o,r,l in zip(samples, outs, raws, losses))
    for i,l in enumerate(["target", "predicted", "probabilities", "loss"]):
        rows = [b.show(ctx=r, label=l, channel=channel, indices=indices, anatomical_plane=anatomical_plane, **kwargs) for b,r in zip(outs.itemgot(i),rows)]
    display_df(pd.DataFrame(rows))