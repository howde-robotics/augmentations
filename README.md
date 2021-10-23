# augmentations

Augmentations are included in the `config.yaml` file that is stored with the zipped data sets. This file also includes the hyperparameters for the network training. This doc will just be talking about the augmentations part of that config file, the functionality of which are located in this repo.

## Strucure

AUgmentations are stored in the augmentations list in the configuratoin `yaml` file. If there isn't already one in the config file you plan to use, just add it yourself. An example of a single augmenttion is shown here.

```yaml
augmentations:
  - type: 'random_erase'
    target_datasets: ['thermal_indoor.tar.xz'] 
    copy: True
    probability: 0.5
    args: []
    kwargs: {'mode': 'object'}
  - <Augmentation #2>
```

Each augmentation is an dictionary element of this list and had 6 mandatry keys:

1. `type`: This string specifies which augmentation is to be applied. The currenly available ones are `ranndom_erase` and `rotate_90`, wich are described below

2. `target_datasets`: This specifies to which datasets this augmentation will apply to. **An empty list will apply the augmentation to all datasets**. The datasets should match those named in the `config.yaml` files `datasets` list. The extensions for the datasets are optional (i.e. `rgb_data.7z` and `rgb_data` will both apply augmentations to the same dataset).

3. `copy`: This specifies if the original images are deleted or kept once an augmentation has been applied. `True` will keep the originals, `False` will delete them.

4. `probability`: This float specifies the probability this augmentation will apply to any given image in the dataset, sampled without replacement *for only this augmentation*. Once an augmentation has been fully applied, all augmented images are aded back to the temporarily stored dataset in Colab. Thus, augmentations specified in this configuration file compound (more below)

4. `args`: This is a list of in-order arguments that will be forwarded to this augmentation function. `[]` will forward none

5. `kwargs`: This is a dict of keyword arguments that will be forwarded to this augmentation function. `{}` will forward none

## Current augmentations

There are currently two augmentations, but making new ones is easy (see below).

### `rotate_90`

The image is rotate clockwise or counter-clockwise 90 degrees (with equal probability). I would put an image here but you know exactly what that means.

### `random_erase`

A rectangular region within the image is deleted and filled with grayscale random noise. See [this paper](https://arxiv.org/abs/1708.04896) for specifics about the algorithm. There are several parameters for this function, most of which specify the qualities of the erased region. The most important parameter is the `mode`. There are three modes of random erase:

- `image`: A single random region in the image is selected and erased. This region is selected irrespective of the objects in the image.
- `object`: For each bounding box in the image, a region is selected and erased.
- `image-object`: Both `image` and `object` level erasure are applied to the same image.



## Order and compounding

## Writing your own augmentation
