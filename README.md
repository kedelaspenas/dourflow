# dourflow: Keras implementation of YOLO v2 

**dourflow** is a keras/tensorflow implementation of the state-of-the-art object detection system [You only look once](https://pjreddie.com/darknet/yolo/). 

- Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
- Github repo: [Darknet](https://github.com/pjreddie/darknet)
 
 
<p align="center">
<img src="result_plots/drivingsf.gif" width="600px"/>
</p>

### Dependancies
---
- [keras](https://github.com/fchollet/keras)
- [tensorflow](https://www.tensorflow.org/)
- [numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/)
- [opencv](https://pypi.org/project/opencv-python/)
- [python 3](https://www.python.org/)

### Usage
---
Running `python3 dourflow.py --help`:

```bash
dourflow: a keras YOLO V2 implementation.

positional arguments:
  action                what to do: 'train', 'validate' or pass an image
                        file/dir.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to input yolo v2 keras model
  -c CONF, --conf CONF  path to configuration file
  -t THRESHOLD, --threshold THRESHOLD
                        detection threshold
  -w WEIGHT_FILE, --weight_file WEIGHT_FILE
                        path to weight file

```
##### *action*
Pass what to do with dourflow:

1. A path to an image file/dir or video: Run inference on those file(s).
2. 'validate': Perform validation on a trained model.
3. 'train': Perform training on your own dataset.

##### *model*
Pass the keras input model h5 file (could be to perform inference, validate against or for transfer learning). 

Pretrained COCO/VOC keras models can be downloaded [here](https://drive.google.com/open?id=1bc_kyb_wpOedHAXruj_TN5uIr7D4D_mc). Alternatively, you can download the weights from [here](https://pjreddie.com/darknet/yolov2/) and generate the model file using [YAD2K](https://github.com/allanzelener/YAD2K).
  

##### *conf*
Pass a config.json file that looks like this (minus the comments!):

```
{
    "model" : {
        "input_size":       416, #Net input w,h size in pixels
        "grid_size":        13, #Grid size
        "true_box_buffer":  10, #Maximum number of objects detected per image
        "iou_threshold":    0.5, #Intersection over union detection threshold
        "nms_threshold":    0.3 #Non max suppression threhsold
    },
    "config_path" : {
        "labels":           "models/coco/labels_coco.txt", #Path to labels file
        "anchors":          "models/coco/anchors_coco.txt", #Path to anchors file
        "arch_plotname":    "" #Model output name (leave empty for none, see result_plots/yolo_arch.png for an example)
    },
    "train": {
        "out_model_name":   "", #Trained model name (saved during checkpoints)
        "image_folder":     "", #Training data, image directory
        "annot_folder":     "", #Training data, annotations directory (use VOC format)
        "batch_size":       16, #Training batch size
        "learning_rate":    1e-4, #Training learning rate
        "num_epochs":       20, #Number of epochs to train for
        "object_scale":     5.0 , #Loss function constant parameter
        "no_object_scale":  1.0, #Loss function constant parameter
        "coord_scale":      1.0, #Loss function constant parameter
        "class_scale":      1.0, #Loss function constant parameter
        "verbose":          1 #Training verbosity
    },
    "valid": {
        "image_folder":     "", #Validation data, image directory
        "annot_folder":     "", #Validation data, annotation directory
        "pred_folder":      "", #Validation data, predicted images directory (leave empty for no predicted image output)
    }
}
``` 
##### *threshold*

Pass the confidence threshold used for detection (default is 30%).

##### *weight_file*

Pass the h5 weight file if generating the YOLO v2 input model. (not needed)

### Inference
---
##### Single Image/Video
Will generate a file in the same directory with a '_pred' name extension. Example:
```bash
python3 dourflow.py theoffice.png -m coco_model.h5 -c coco_config.json -t 0.35
```
<p align="center">
<img src="result_plots/theoffice.png" width="600"/>
<img src="result_plots/theoffice_pred.png" width="600px"/>
</p>

##### Batch Images
Will create a directory named **out/** in the current one and output all the images with the same name.

Example:
```bash
python3 dourflow.py theoffice.png -m coco_model.h5 -c coco_config.json -t 0.35
```
<p align="center">
<img src="result_plots/batchex.png" width="500px"/>
</p>


### Validation
---
Allows to evaluate the performance of a model by computing its [mean Average Precision](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000) in the task of object detection (mAP WRITE UP COMING SOON).

Example:
```bash
python3 dourflow.py validate -m voc_model.h5 -c voc_config.json
```
Terminal output:
```bash
Batch Processed: 100%|████████████████████████████████████████████| 4282/4282 [01:53<00:00, 37.84it/s]
AP( bus ): 0.806
AP( tvmonitor ): 0.716
AP( motorbike ): 0.666
AP( dog ): 0.811
AP( horse ): 0.574
AP( boat ): 0.618
AP( sofa ): 0.625
AP( sheep ): 0.718
AP( bicycle ): 0.557
AP( cow ): 0.725
AP( pottedplant ): 0.565
AP( train ): 0.907
AP( bird ): 0.813
AP( person ): 0.665
AP( car ): 0.580
AP( cat ): 0.908
AP( bottle ): 0.429
AP( diningtable ): 0.593
AP( chair ): 0.475
AP( aeroplane ): 0.724
-------------------------------
mAP: 0.674

```


### Training
---
**NEED TO DO A FEW FIXES TO LOSS FUNCTION BEFORE THIS IS DONE**
---



##### Split dataset
Script to generate training/testing splits.

`python3 split_dataset.py -p 0.75 --in_ann VOC2012/Annotations/ --in_img VOC2012/JPEGImages/ --output ~/Documents/DATA/VOC`


##### Tensorboard

Training will create directory **logs/** which will store loss and checkpoints for all the different runs during training.
 
Model passed is used for [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) (TRAINING FROM SCRATCH / TRAINING ONLY LAST LAYER SHOULD BE ADDED SOON).

Example:
`python3 dourflow.py train -m models/logo/coco_model.h5 -c confs/config_custom.json`

Then, in another terminal tab you can run `tensorboard --logdir=logs/run_X` and open a browser page at `http://localhost:6006/` to monitor the train/val loss:

<p align="center">
<img src="result_plots/tbexam.png" width="600px"/>
</p>




#### To Do

- [ ] TRAINING BUG / FINISH LOSS FUNCTION
- [ ] cfg parser
- [ ] Anchor generation for custom datasets
- [ ] mAP write up


#### Inspired from

- [Darknet](https://github.com/pjreddie/darknet)
- [Darkflow](https://github.com/thtrieu/darkflow)
- [keras-yolo2](https://github.com/experiencor/keras-yolo2)
- [YAD2K](https://github.com/allanzelener/YAD2K)