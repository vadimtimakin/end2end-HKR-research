# A modern approach to the end-to-end bilingual handwriting text recognition on the example of russian school notebooks

## Abstract
In repository we provide our approach to the end-to-end bilingual handwriting text recognition. Our solution was developed within the framework of the NTO AI Olympics 2022 and was awarded a prize of $10,000. The dataset presented in the competition consisted of school notebooks, which contained texts in Russian and English. The goal was to first detect the words on the sheet and then recognize the text in them.

## Pipeline

<img src="https://github.com/t0efL/end2end-HKR-research/blob/main/images/pipeline.jpg" alt="drawing" width="450"/>

## OCR part

### Handling bilingual texts
There are two main approaches to the recognition of multilingual texts: with a single model and using multiple models for each of the languages and an additional classification model for determining the language. We tried them both and came to the conclusion that it's better to use a singe model in this task. Within the framework of our experiments, a single model shows a relatively good result compared to separate models. Also equally important is the fact that using a single model reduces the time and memory required for inference, which makes it more suitable for a product solution.

### Validation strategy
Our validation strategy for OCR part is a Stratified K-fold split based on the both `text lengths` and `characters occurance`. We were forced to use only a single fold on inference due to the time limits. We split the data into 10 folds and took one of them (train 90%, validation 10%).

### Modeling and loss functions

<img src="https://github.com/t0efL/end2end-HKR-research/blob/main/images/model.jpg" alt="drawing" width="450"/>

#### Architecture
After testing CRNN and Transformer architectures separately we decided to create one `gybrid model` with tho heads: CTC's one and Transformer's one. This move allowed us to achieve maximum perfomance. In the first experiments, the two separate architectures mentioned above showed approximately similar results, so simply combining them into one does not guarantee an increase in accuracy. Instead, improving the quality of the combined model is a consequence of training it for two loss functions at once: `CTC Loss` and `Cross Entropy Loss`. During the experiments, we were able to choose the best weights for each of the losses: `0.25` and `0.75` accordingly.

#### Backbone (feature extractor)
We've tested a list of different backbones. It's necessarily to always test a backbone with different optimizers. We started from the Resnet + Adam bundle which was a good baseline. Ranger optimizer improved the accuracy of Resnet significantly. During our work on this task the Convnext has been released. It showed a low perfomance with Ranger optimizer, slightly better one with Adam and extemely good with MADGRAD (MADGRAD didn't work with Resnet). We've also tested different backbone size. Everything that was greater than "base" size was overfitting and didn't show a good perfomance. The "base" size has shown the best perfomance, but due to the time limits on inference we were forced to pick `Convnext small + MADGRAD` bundle for our final solution.

#### Prefix / postfix models
After analyzing the most frequent errors of our model, we noticed that many of them were either at the beginning or at the end. We tried to train separate prefix and postfix models and add them to our pipeline, but it didn't improve our score.

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

### Inference

<img src="https://github.com/t0efL/end2end-HKR-research/blob/main/images/inference.jpg" alt="drawing" width="450"/>

#### Merging CTC's and Transformers's heads predictions

#### Speed up

#### Language model

#### Beam Search

## Detection (segmentation) part

## Possible approaches
There are two ways to get the cropped words from the sheet: `detection` and `segmentation`. We tried them both. For each one we've chosen the current SOTA model: `YOLOV5` and `Detectron` respectively.

### Validation strategy
Our validation strategy for detection part is a Stratified K-fold split based on the both `sheet orientation` and `language on a sheet`. We were forced to use only a single fold on inference due to the time limits. We split the data into 2w0 folds and took one of them (train 95%, validation 5%).

### Detection (YOLOV5)

The YOLOV5 model has been trained and used with the following setup:
- Epochs: 500
- Optimizer: AdamW
- Tuned parameters
- Image size: 1280
- Model: yolov5l
- Batchsize: 2
- Lower threshold on inference (0.5 -> 0.45)

### Segmetation (Detectron)

The Detectron model has been trained and used with the following setup:
- Iterations: 20k
- Optimizer: MADGRAD
- Tuned parameters
- Backbone: Convnext
- Increased number of bboxes on inference
- Increased image size on infernce (1024 -> 2048)

### Metric for evaluation

Since the text is recognized on the polygons predicted by the model, in order to understand which text to compare the one predicted by the model, it is necessary to correlate the predicted polygons with ground truth polygons.

The metric script searches for the predicted polygon corresponding to each ground truth polygon from the notebook. From the predicted polygons, the one that has the `largest IoU intersection` with the ground truth polygon is selected (in this case, the IoU must be greater than zero). Thus, the ground truth text from this polygon correlates with the predicted text. This is a `true positive` prediction.

`False negative` cases: if the predicted polygon was not matched for a ground truth polygon, then the predicted text for such a polygon is set as empty "" (because the pipeline did not predict the text where it should be).

For all `false positive` predicted polygons (i.e. those for which there are no ground truth polygons), the ground truth text is set as empty "" (because the pipeline predicted the text where there is none).

Finally, `micro CER` is calculated for each image and then averaged for all images.

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

## Conclusion
As a result of our  work we managed to:
1) Achieve 3.2 CER for bilingual OCR model.
2) Achive 13.87 CER overall.
3) Figure out that the detection approach based on the YOLOV5 is more efficient than the segmentation apporoach based on the Detectron.

## Team
- Vadim Timakin [GitHub](https://github.com/t0efL) | vad.timakin@yandex.ru | [Telegram](https://t.me/mrapplexz)
- Maxim Afanasyev [GitHub](https://github.com/mrapplexz) | mr.applexz@gmail.com | [Telegram](https://t.me/t0efL)
