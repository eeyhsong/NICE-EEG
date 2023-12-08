# Decoding Nature Images from EEG for Object Recognition [[Paper](https://arxiv.org/abs/2308.13234)]
Natural Image Contrast EEG (NICE)

## Abstract
![Network Architecture](/visualization/Fig1.png)
Electroencephalography (EEG) signals, known for convenient non-invasive acquisition but low signal-to-noise ratio, have recently gained substantial attention due to the potential to decode natural images. This paper presents **a self-supervised framework** to demonstrate the feasibility of learning image representations from **EEG** signals, particularly for **object recognition**. The framework utilizes image and EEG encoders to extract features from paired image stimuli and EEG responses. Contrastive learning aligns these two modalities by constraining their similarity. With the framework, we attain significantly above-chance results on a comprehensive EEG-image dataset, achieving a top-1 accuracy of 15.6% and a top-5 accuracy of 42.8% in challenging 200-way zero-shot tasks. Moreover, we perform extensive experiments to explore the **biological plausibility by resolving the temporal, spatial, spectral, and semantic aspects of EEG signals**. Besides, we introduce **attention modules** to capture spatial correlations, **providing implicit evidence** of the brain activity perceived from EEG data. These findings yield valuable insights for neural decoding and brain-computer interfaces in real-world scenarios.

## Datasets
many thanks for sharing good datasets!
1. [Things-EEG2](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)
2. [Things-MEG](https://elifesciences.org/articles/82580) (updating)

## Pre-processing
### Script path
- ./preprocessing/
### Data Path 
- raw data: ./Data/Things-EEG2/Raw_data/
- proprocessed eeg data: ./Data/Things-EEG2/Preprocessed_data_250Hz/
### Steps
1. pre-processing EEG data of each subject
   - modify `preprocessing_utils.py` as you need.
     - choose channels
     - epoching
     - baseline correction
     - resample to 250 Hz
     - sort by condition
     - **Multivariate Noise Normalization**
   - `python preprocessing.py` for each subject. 

2. get the center images of each test condition (for testing, contrast with EEG features)
   - get images from original Things dataset but discard the images used in EEG test sessions.
  
## Get the Features from Pre-Trained Models
### Script path
- ./dnn_feature_extraction/
### Data Path (follow the original dataset setting)
- raw image: ./Data/Things-EEG2/Image_set/image_set/
- preprocessed eeg data: ./Data/Things-EEG2/Preprocessed_data/
- features of each images: ./Data/Things-EEG2/DNN_feature_maps/full_feature_maps/model/pretrained-True/
- features been packaged: ./Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/model/pretrained-True/
- features of condition centers: ./Data/Things-EEG2/Image_set/
### Steps
1. obtain feature maps with each pre-trained model with `obtain_feature_maps_xxx.py` (clip, vit, resnet...)
2. package all the feature maps into one .npy file with `feature_maps_xxx.py`
3. obtain feature maps of center images with `center_fea_xxx.py`
   - save feature maps of each center image into `center_all_image_xxx.npy`
   - save feature maps of each condition into `center_xxx.npy` (used in training)

## Training and Testing
### Script path
- `./nice_stand.py`

## Visualization - updating
### Script path
- ./visualization/
### Steps

## Milestones
1. nice_v0.50 NICE (natural image contraste eeg)

## Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊
```
@misc{song2023decoding,
  title = {Decoding {{Natural Images}} from {{EEG}} for {{Object Recognition}}},
  author = {Song, Yonghao and Liu, Bingchuan and Li, Xiang and Shi, Nanlin and Wang, Yijun and Gao, Xiaorong},
  year = {2023},
  month = nov,
  number = {arXiv:2308.13234},
  eprint = {2308.13234},
  primaryclass = {cs, eess, q-bio},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2308.13234},
  archiveprefix = {arxiv}
}
```
<!-- ## Acknowledgement

## References

## License -->

