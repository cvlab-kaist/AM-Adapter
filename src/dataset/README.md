<h1> Dataset Preparation  </h1>

1. Download BDD100K dataset from the following url :  https://dl.cv.ethz.ch/bdd100k/data/ 

    The required files are:
    | Dataset | Url |
    |------|---|
    | 10k_images_train.zip | [Download](https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip)|
    | 10k_images_val.zip | [Download](https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip) |
    | 10k_images_test.zip | [Download](https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip) |
    | bdd100k_sem_seg_labels_trainval.zip | [Download](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip) |

2. Unzip each zip file and organize them as follows: 
    ```bash
    ├── bdd100k
    ├──── images
    ├────── 10k 
    ├──────── test
    ├──────── train 
    └──────── val  
    ├──── labels
    ├────── 10k
    ├──────── sem_seg
    ├────────── colormaps
    ├────────── masks
    ├────────── polygons
    └────────── rles
    ```

3. Download JSON file for retrieval 
    | Dataset | Url |
    |------|---|
    | BDD100K retrieved pair json | [Download](https://drive.google.com/file/d/18CUXcBvbZJvNCH5wO2dZmoIT-5ReF8gc/view?usp=share_link) |
