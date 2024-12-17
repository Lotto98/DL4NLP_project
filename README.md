# DL4NLP Project

## How to Install

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd DL4NLP_project
    ```
    Replace `<repository_url>` with the actual URL of the repository.

2. Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

3. Create the conda environment:
    ```sh
    conda env create -f environment.yml
    ```

4. Activate the conda environment:
    ```sh
    conda activate DL4NLP
    ```

5. Install Detectron2:
    ```sh
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

## Create the dataset

### AMI dataset

To create the AMI dataset, follow these steps:

1. Run the `dataset_creation_AMI.py` script to create annotations and download audio files:
    ```sh
    python3 dataset_creation_AMI.py --split all
    ```
    This will create annotations and download audio files for the train, validation, and test splits.

2. The annotations will be saved in the `datasets/ami/annotations` directory and the audio files will be saved in the `datasets/ami/audio` directory.

## Train the Model (Optional)

To train the model on the AMI dataset using the AST backbone, run:
```sh
python3 train_net.py --num-gpus n --config-file configs/diffdet.ami.ast.yaml
```

## Inference

To perform inference, run:
```sh
python3 ...
```

## Additional Information

For more details, refer to the [documentation](docs/documentation.md).