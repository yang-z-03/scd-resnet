
# Gaussian clustering and quantiﬁcation of sperm chromatin dispersion test using convolutional neural network

## System Requirements

the version status of the training environment is list below:
* conda 4.8.2
* pytorch 1.7.1
* torchvision 0.8.2
* python 3.8, 3.8.12

## How to

### Annotate SCD Images

SCD images should be captured with 20x objective with auto white balance. To obtain the valid annotations for these images, you can take use of `SCD Label` utility software. It is a C# GUI application that runs only on Windows.

1. First, build the `SCD Label` software with Visual Studio. 
2. Place all your images under one folder, say, `C:\folder-images\`.
3. By clicking `Open image directory ...` on the toolstrip, select `folder-images` and a list of images to annotate (in the correct format) will be displayed in the list box on the left.
4. Then, click `Open save directory ...` and specify the folder to output annotation marks, which is in the format of plain text. The annotation is saved to the `.txt` file with the same name as the image file.
5. Select one image in the list and begin annotating, there is a video demonstrating how to use the software in electronic supplementary materials. You are able to access it once the article is published.
6. Go through all the images, and now you have one folder for the annotation, and another for the original images.


### Preprocess the Images

The preprocess script can convert the original images (of very large image size) to unified 512 x 512 image clips with 16x random rotation augmentation. You can choose some of the given random rotation ratio, as well as to use a percentage of the training data. These are provided with several preprocess presets `datasets.scds.scdx__p__`.

```
usage: preprocess.py [-h] [-i INPUTIMAGE] [-a ANNOTATION] [-s DESTINATIONSIZE] [-t IOUTHRESHOLD] [-v]
                     [-m MARGIN] [-p PROFILE]
                     outputZipPath

preprocess.py - sample preprocess executable for neural network training: raw full slide images will be
clipped and transformed into a specified size and labelimg format of annotations will be decoded to
corresponding heatmap in the form of numpy array savings.

positional arguments:
  outputZipPath       the location to place the output zipped samples.

optional arguments:
  -h, --help          show this help message and exit
  -i INPUTIMAGE       input image folder.
  -a ANNOTATION       input annotation folder in labelImg YOLO format.
  -s DESTINATIONSIZE  destination image size.
  -t IOUTHRESHOLD     IoU threshold for gaussian radius determination.
  -v                  display the heatmap and clip result (debug).
  -m MARGIN           the border margin to fill blank, in the form of 'leftMargin topMargin rightMargin    
                      bottomMargin'.
  -p PROFILE          the preprocess profile module
```

```bash
python preprocess.py -i "<path_to_dataset>/folder-images/" -a "<path_to_dataset>/folder-annotations/" -s 512 -t 0.5 -m "0 0 0 0" -p "datasets.scds.scdx16p100" "<dataset>.d"
```

### Training

Once you have generated a `*.d` dataset file, you can proceed training the network. Create a JSON configuration with the following:

```json
exp.json:

{
    "datasetName": "scdx16p100",            // the dataset preprocess profile
    "modelName": "centerOffsetRes10",       // your model name, corresponding the python file in trainer/model/
    "trainName": "arbitary-train-name-xxx", // the name of your output file, arbitary

    // the following are training parameters
    "batchSize": 32,
    "validation": 2,
    "validationBatchSize": 64,
    "iterations": 3,
    "snapshot": 3,
    "learningRate": 0.000125,
    "learningRateDecay": [11000],
    "learningRateDecayRate": [10],


    "dirTemp": "/temp/",
    "dirResult": "/results/",
    "dirConfig": "/src/configs/",

    // where you place the **.d data file. it should be placed at "{dirDataset}{datasetName}.d"
    "dirDataset": "/data/"
}
```

```bash
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /configs/exp.json -gpu
```

### Obtaining the Result

See the example in `test.py`

### Want a Pre-trained Model For Testing?

You have one, see `pretrained/model70.pt`. See `test.py`, and there is a place to use that model. You get `.pth` files by training the network, and you may use

```bash
python trace.py -a <{modelName} in your config> -m <your-model-file.pth> -s '1 1 512 512' -wrapped output.pt
```

# Supplementary Materials

DOI:	https://doi.org/10.1039/D3AN01616A
