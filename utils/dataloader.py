import torch
from torchinfo import summary

import torchvision
from torchvision import transforms as tf


CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx


def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'


class VOCTransform:
    def __init__(self, train=True, only_person=False):
        self.only_person = only_person
        self.train = train
        if train:
            self.augmentation = tf.RandomApply([tf.ColorJitter(0.2, 0.2, 0.2, 0.2)])

    def __call__(self,image, target):
        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = tf.functional.resize(image, size=(new_height, new_width))
        image = tf.functional.pad(image, padding = (diff_width//2,
                                                            diff_height//2,
                                                            diff_width//2 + diff_width % 2,
                                                            diff_height//2 + diff_height % 2))
        target = target['annotation']['object']

        target_vectors = []
        for item in target:
            x0 = int(item['bndbox']['xmin'])
            x1 = int(item['bndbox']['xmax'])
            y0 = int(item['bndbox']['ymin'])
            y1 = int(item['bndbox']['ymax'])

            target_vector = [(diff_width/2 + (x0 + x1)/2) / (img_width + diff_width),
                    (diff_height/2 + (y0 + y1)/2) / (img_height + diff_height),
                    (max(x0, x1) - min(x0, x1)) / (img_width + diff_width),
                    (max(y0, y1) - min(y0, y1)) / (img_height + diff_height),
                    1.0,
                    class_to_num(item['name'])]

            if self.only_person:
                if target_vector[5] == class_to_num("person"):
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        if self.train:
            return self.augmentation(tf.functional.to_tensor(image)), target_vectors
        else:
            return tf.functional.to_tensor(image), target_vectors


def VOCDataLoader(train=True, batch_size=32, shuffle=False):
    if train:
        image_set = "train"
    else:
        image_set = "val"

    dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=True,
                                                transforms=VOCTransform(train=train))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def VOCDataLoaderPerson(train=True, batch_size=32, shuffle=False):
    if train:
        image_set = "train"
    else:
        image_set = "val"
    
    dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=True,
                                                transforms=VOCTransform(train=train, only_person=True))
    indices = [i for i in range(len(dataset)) if torch.any(dataset[i][1][:,-1] == 0)]
    dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)