# Learning and Evaluation of Visual Discovery Models with Incomplete Labels
## Submission code

This repository contains the implementation of the main features of the paper. the full code and dataset will be released upon acceptance.

## Code repository files
1. tcl_loss.py - contains the Loss used for TCL finetuning.
2. metrics.py - contains the metrics used for our experiments.
3. transitive_labels_graph.py - contains the GT graph creation + transitive labels deduction.
4. transitive_batch_creation.py - contains the logic we used to calculate each batch's label matrix.


## VSD Furniture Dataset
We provide the GT annotation of the VSD Furniture dataset.

### Annotations file
The annotations file can be found at:
https://drive.google.com/drive/folders/1uiYjtwZCeYo19C-1whXmqO05qpXxRYsr?usp=drive_link

The file contains a json in the following structure:
```json
    [
        {
            "key": ["img/image_001.jpg", "img/image_002.jpg"],
            "value": 0,
        },
        {
            "key": ["img/image_002.jpg", "img/image_008.jpg"],
            "value": 1,
        },
        {
            "key": ["img/image_001.jpg", "img/image_004.jpg"],
            "value": 1,
        }
    ]
```

Each `key` is an array where the first image path is the query (seed), and the second is the gallery (candidate).
The `value` field contains the ground truth label, 1 for a positive pair, and 0 for a negative pair.


### Images
The dataset images are accessible at (using the furniture class):
https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

The images should be placed under a folder named `img` to match the annotations file.

