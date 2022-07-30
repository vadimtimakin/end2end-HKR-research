# A modern approach to the end-to-end bilingual handwriting text recognition on the example of russian school notebooks

## Abstract
In repository we provide our approach to the end-to-end bilingual handwriting text recognition. Our solution was developed within the framework of the NTO AI Olympics 2022 and was awarded a prize of $10,000. The dataset presented in the competition consisted of school notebooks, which contained texts in Russian and English. The goal was to first detect the words on the sheet and then recognize the text in them.

## OCR part

### Validation strategy
Our validation strategy is a Stratified K-fold split based on the both `text lengths` and `characters occurance`.

### Modeling and losses

### Training setup
- Epochs: 55 (50 + 5 warm-up)
- Optimizer: MADGRAD
- Learning rate: 1e-4
- Scheduler: Cosine (T_max=50)
- Mixed Precision: OFF (doesn't work with CTC Loss)
- Gradient accumulation: OFF 
- Batchnorms freeze: OFF
- Gradient Clipping: ON (2)
- Batch size: 32

### Pre-processing
- Custom resize function with saving text aspect ratio using padding (384 x 96)
- Normalization (imagenet)
- Swap black background to white one
- Convert image to grayscale

### Augmentations
- Custom augmentation for artificial crossing out of letters
- Blur
- CLAHE 
- Rotate 
- Cutout 
- GridDistortion 
- RandomShadow
- MotionBlur
- Optical Distortion
- Turned off augmentations during the warmup  

### Datasets

### Post-processing

## Detection part

### Validation strategy

### YOLOV5

### Detectron

### Comparing

## Overall results

## Team
