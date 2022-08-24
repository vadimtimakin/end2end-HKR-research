import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from PIL import ImageDraw, Image

from skimage import color


class SmartResizeV2:
    """
    Resizes image avoiding changing its aspect ratio.
    Includes stretching and squeezing augmentations.
    """

    def __init__(self, width, height, stretch=(1, 1), fillcolor=0):
        """
        Args:
        
            width (int): target width of the image.
            
            height (int): target height of the image.
            
            stretch (tuple): defaults to (1, 1) - turned off.
            Parameter for squeezing / stretching augmentation,
            tuple containining to values which represents
            the range of queezing / stretching.
            Values less than 1 compress the image from the sides (squeezing),
            values greater than 1 stretch the image to the sides (stretching).
            Use range (1, 1) to avoid squeezing / stretching augmentation.
            
            fillcolor (int): defults to 255 - white. Number in range [0, 255]
            representing fillcolor.
        """

        assert len(
            stretch) == 2, "stretch has to contain only two values " \
                           "representing range. "
        assert stretch[0] >= 0 and stretch[
            1] >= 0, "stretch has to contain only positive values."
        assert 0 <= fillcolor <= 255, "fillcolor has to contain values in " \
                                      "range [0, 255]. "

        self.width = int(width)
        self.height = int(height)
        self.stretch = stretch
        self.ratio = int(width / height)
        self.color = fillcolor

    def __call__(self, img):
        """
        Transformation.
        Args:
            img (np.array): RGB np.array image which has to be transformed.
        """

        stretch = random.uniform(self.stretch[0], self.stretch[1])

        img = np.array(img)
        h, w, _ = img.shape
        img = cv2.resize(img, (w, int(w / (w * stretch / h))))
        h, w, _ = img.shape

        if not (w / h) == self.ratio:
            x = random.random()
            if (w / h) < self.ratio:
                base = self.ratio * h - w
                white_one = np.zeros([h, int(base * x), 3], dtype=np.uint8)
                white_one.fill(self.color)
                white_two = np.zeros([h, int(base * (1 - x)), 3], dtype=np.uint8)
                white_two.fill(self.color)
                img = cv2.hconcat([white_one, img, white_two])
            elif (w / h) > self.ratio:
                base = (w - self.ratio * h) // self.ratio
                white_one = np.zeros([int(base * x), w, 3], dtype=np.uint8)
                white_one.fill(self.color)
                white_two = np.zeros([int(base * (1 - x)), w, 3], dtype=np.uint8)
                white_two.fill(self.color)
                img = cv2.vconcat([white_one, img, white_two])
        img = cv2.resize(img, (self.width, self.height))

        return img


class ExtraLinesAugmentation:
    '''
    Add random black lines to an image
    Args:
        number_of_lines (int): number of black lines to add
        width_of_lines (int): width of lines
    '''

    def __init__(self, number_of_lines: int = 2, width_of_lines: int = 8):
        self.number_of_lines = number_of_lines
        self.width_of_lines = width_of_lines
      
    def __call__(self, img):
        '''
        Args:
          img (PIL Image): image to draw lines on
        Returns:
          PIL Image: image with drawn lines
        '''
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for _ in range(self.number_of_lines):
            x1 = random.randint(0, np.array(img).shape[1]); y1 = random.randint(0, np.array(img).shape[0])
            x2 = random.randint(0, np.array(img).shape[1]); y2 = random.randint(0, np.array(img).shape[0])
            draw.line((x1, y1, x2 + 100, y2), fill=0, width=self.width_of_lines)

        return np.array(img)


transforms = A.Compose([
    # A.Rotate(3, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    # A.CLAHE(clip_limit=1.5, p=1.0),
    # A.Cutout(num_holes=10, p=0.5),
    # A.GridDistortion(distort_limit=0.15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    # A.Blur(blur_limit=1.5, p=1.0),
    # A.MotionBlur(blur_limit=(6, 6), p=1.0),
    # A.RandomShadow(p=0.5),
])

# r = A.OpticalDistortion(distort_limit=(1.0, 1.0), p=1.0)

path = '/home/toefl/K/nto/x_dataset/data/train_recognition/images'
for file in os.listdir(path)[100:200]:
    img = cv2.imread(os.path.join(path, file))

    plt.imshow(img)
    plt.show()

    c1, c2, c3 = img.T
    img[...][((c1 == 0) & (c2 == 0) & (c3 == 0)).T] = (255, 255, 255)
    # img = r(image=img)["image"]
    resize = SmartResize(384, 96, stretch=(1.0, 1.0), fillcolor=255)
    lines = ExtraLinesAugmentation(2, 7)
    img = resize(img)
    # img = lines(img)
    img = transforms(image=img)["image"] / 255
    # img = color.rgb2gray(img)

    plt.imshow(img)
    plt.show()