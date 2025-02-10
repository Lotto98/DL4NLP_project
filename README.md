# DL4NLP Project

Final project of the DL4NLP course.

## How to Install

1. Clone the repository:
    ```sh
    git clone https://github.com/Lotto98/DL4NLP_project
    cd DL4NLP_project
    ```

2. Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

3. Create the conda environment:
    ```sh
    conda env create -f environment_yolo.yml
    ```

4. Activate the conda environment:
    ```sh
    conda activate DL4NLP_yolo
    ```

## Prepare dataset (AMI)

1. Download the dataset:
    ```sh
    python3 dataset_creation_AMI.py --split all
    ```

2. Prepare the data for Yolo training/inference
    ```sh
    python3 yolo/yolo_dataset.py --split all
    ```

## Train the YOLO model
```bash
usage: yolo_training.py [-h] 
--model {yolo11n.pt,yolo11s.pt,yolo11m.pt,yolo11l.pt,yolo11x.pt,resume} 
--batch BATCH 
--epochs EPOCHS 
--imgsz IMGSZ 
[--path-model PATH_MODEL]
```

| Option                            | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
|  `-h, --help`                     | Show this help message and exit.                            |
|  `--model MODEL`                  | Yolo model to train. If resume is passed then also `--path-model PATH_MODEL` is required.  |
|  `--batch BATCH`                  | Batch size to use.                                          |
|  `--epochs EPOCHS`                | Number of training epochs.                                  |
|  `--imgsz IMGSZ`                  | Image size to use.                                          |
|  `--path-model PATH_MODEL`        | Model path to use (only required if `--model resume`)       |

## Test the model

```sh
usage: yolo_test.py [-h] [--name {nano,medium}] [--image-size {640,1216}]
```
| Option                            | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
|  `-h, --help`                     | Show this help message and exit.                            |
|  `--name MODEL`                   | Model name to load: nano or medium.                         |
|  `--image-size IMAGE-SIZE`        | Image size of the model to load.                            |

## Whisper inference with YOLO

```sh
usage: post_processing.py [-h] [--model-name {nano,medium}] [--image-size {640,1216}]
```

| Option                            | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
|  `-h, --help`                     | Show this help message and exit.                            |
|  `--name MODEL`                   | Model name to load: nano or medium.                         |
|  `--image-size IMAGE-SIZE`        | Image size of the model to load.                            |