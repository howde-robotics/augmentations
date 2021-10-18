import os
import yaml
import random 
import PIL

import augmentations.augmentations
import augmentations.augment_util

def _getTrainingConfig(training_directory : str, config_filename : str):
  if not config_filename.endswith('.yaml'):
    config_filename += '.yaml'
  network_config_filename = os.path.join(training_directory, config_filename)
  with open(network_config_filename, "r") as stream:
      try:
          network_config = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc)
  return network_config

def applySingleAugmentation(apply_to, augment_func, augment_type, augment_copy, 
                    augment_probability, augment_args, augment_kwargs):

    for target_set in apply_to:
        training_dir = os.path.join(target_set, 'train')
        print(f"Apply augmentation \'{augment_type}\' to data in {training_dir}")
        print(f"\tp={augment_probability}, copy={augment_copy}")
        print(f"\targs={augment_args}, kwargs={augment_kwargs}")

        #list if image and txt files, with same names 
        # AAA.jpeg and AAA.txt for example
        original_training_list = os.listdir(training_dir)

        #only want unique names, each entry has img and txt associated with it
        original_name_list = list(set([os.path.splitext(s)[0] for s in original_training_list]))

        num_to_augment = int(augment_probability * len(original_name_list))
        random.shuffle(original_name_list)
        to_augment = original_name_list[:num_to_augment]

        for name in to_augment:
            orig_img_file, orig_annotation_file = augment_util.findRelevantTrainingEx(original_training_list, name)
            orig_img_path = os.path.join(training_dir, orig_img_file)
            orig_annotation_path = os.path.join(training_dir, orig_annotation_file)

            #apply augmentation and forward arguments
            pil_img, annotation_list = augment_func(orig_img_path, orig_annotation_path, 
                                                    *augment_args, **augment_kwargs)

            #save augmentation
            new_img_path = os.path.join(training_dir, f'{augment_type}_{orig_img_file}')
            new_annotation_path = os.path.join(training_dir, f'{augment_type}_{orig_annotation_file}')
            pil_img.save(new_img_path)
            with open(new_annotation_path, 'w') as f:
                for anno in annotation_list:
                    f.write(f"{anno[0]} {anno[1]} {anno[2]} {anno[3]} {anno[4]}\n")

            #maybe delete original 
            if not augment_copy:
                #delete old versions
                os.remove(orig_img_path)
                os.remove(orig_annotation_path)

def applyAugmentations(config, local_storage_dir, local_dsets_list):
    augments_to_apply = config['augmentations']

    print(__name__)

    # target_augment = augments_to_apply[0]
    for target_augment in augments_to_apply:
        augment_type = target_augment['type']
        augment_copy = target_augment['copy']
        augment_probability = target_augment['probability']
        augment_args = target_augment['args']
        augment_kwargs = target_augment['kwargs']

        apply_to = augment_util.findTargetDatasets(target_augment, local_storage_dir, local_dsets_list)

        if augment_type == 'random_erase':
            augment_func = augmentations.augment_random_erase
        elif augment_type == 'rotate_90':
            augment_func = augmentations.augment_rotate_90
        else:
            raise NotImplementedError(f'Augmentation type \'{augment_type}\' not supported')

        applySingleAugmentation(apply_to, augment_func, augment_type, augment_copy, 
                        augment_probability, augment_args, augment_kwargs)

def main():
    local_storage_dir = './'
    local_dsets_list = ['./thermal_indoor_playground_small',
                        './big_ol_fake']

    this_config = _getTrainingConfig('./', 'thermal_augmentation_test.yaml')
    applyAugmentations(this_config, local_storage_dir, local_dsets_list)
    

if __name__ == '__main__':
    main()


        