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

Each augmentation is an dictionary element of this list and had 5 mandatry keys:

1. `type`: This string specifies which augmentation is to be applied. The currenly available ones are `ranndom_erase` and `rotate_90`, wich are described below

2. `target_datasets`: This specifies to which datasets this augmentation will apply to. **An empty list will apply the augmentation to all datasets**. The datasets should match those named in the `config.yaml` files `datasets` list. The extensions for the datasets are optional (i.e. `rgb_data.7z` and `rgb_data` will both apply augmentations to the same dataset).

3. `copy`: This specifies if the original images are deleted or kept once an augmentation has been applied. `True` will keep the originals, `False` will delete them.

4. `args`: This is a list of in-order arguments that will be forwarded to this augmentation function. `[]` will forward none

5. `kwargs`: This is a dict of keyword arguments that will be forwarded to this augmentation function. `{}` will forward none

## Current augmentations

## Order and compounding

## Writing your own augmentation
