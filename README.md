# A modern approach to the end-to-end bilingual handwriting text recognition on the example of russian school notebooks

## Abstract
In repository we provide our approach to the end-to-end bilingual handwriting text recognition. Our solution was developed within the framework of the NTO AI Olympics 2022 and was awarded a prize of $10,000. The dataset presented in the competition consisted of school notebooks, which contained texts in Russian and English. The goal was to first detect the words on the sheet and then recognize the text in them.

## OCR part

### Handling bilingual texts

### Validation strategy
Our validation strategy for OCR part is a Stratified K-fold split based on the both `text lengths` and `characters occurance`. We were forced to use only a single fold on inference due to the time limits. We split the data into 10 folds and took one of them (train 90%, validation 10%).

### Modeling and loss functions

![alt text](https://github.com/t0efL/end2end-HKR-research/blob/main/images/model.png)

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

#### Russian

#### English

### Post-processing

#### Merging CTC's and Transformers's heads predictions

#### Language model

#### Beam Search

## Detection (segmentation) part

## Possible approaches
There are two ways to get the cropped words from the sheet: detection and segmentation. We tried them both. For each one we've chosen the current SOTA model: YOLOV5 and Detectron respectively.

### Validation strategy
Our validation strategy for detection part is a Stratified K-fold split based on the both sheet orientation and language on a sheet. We were forced to use only a single fold on inference due to the time limits. We split the data into 2w0 folds and took one of them (train 95%, validation 5%).

### YOLOV5

The YOLOV5 model has been trained and used with the following setup:
- Epochs: 500
- Optimizer: AdamW
- Tuned parameters
- Image size: 1280
- Model: yolov5l
- Batchsize: 2
- Lower threshold on inference (0.5 -> 0.45)

### Detectron

The Detectron model has been trained and used with the following setup:
- Iterations: 20k
- Optimizer: MADGRAD
- Tuned parameters
- Backbone: Convnext
- Increased number of bboxes on inference
- Increased image size on infernce (1024 -> 2048)

### Metric for evaluation

### Comparing

Here are the finals scores for both models.

| Model | Public LB | Private LB |
| --- | --- | --- | 
| Detectron | 0.1303  | 0.1394 |
| YOLOV5  | **0.1205** | **0.1387** |

Now let's compare these two approaches based on two different models using a list of criteria.

| Criteria | YOLOV5 | Detectron | Winner (D / Y) |
| --- | --- | --- | --- | 
| Tuning needed  | Minor config tunning | Parameter tuning and custom backbone were required | Y | 
| Training time | Slow | Fast | D | 
| Inference time | Lightning fast | Slow | Y | 
| Perfomance | Good | A little worse | Y | 
| The complexity of the annotantions | Only bboxes needed | Annotains for segmentation required | Y | 

## Overall results

## Team
- Vadim Timakin [GitHub](https://github.com/t0efL) | vad.timakin@yandex.ru | [Telegram](https://t.me/mrapplexz)
- Maxim Afanasyev [GitHub](https://github.com/mrapplexz) | mr.applexz@gmail.com | [Telegram](https://t.me/t0efL)
