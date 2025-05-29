# Project 3: CNN for Image Classification


In this project, we have implemented ten notable CNN architectures for image classification on the MNIST dataset. The
architectures include:

- 5 classical CNN architectures: `LeNet-5`, `AlexNet`, `VGG-11`, `GoogLeNet`, and `ResNet-18`.
- 5 lightweight CNN architectures: `SqueezeNet`, `ShuffleNetV2`, `MobileNetV3`, `MNASNet` and `EfficientNet-B0`.

Asides from classification, we use the feature vector extracted from the CNN model to perform clustering and
visualization.

Please note that the models are not 100% authentically reproduced, but adapted to the MNIST dataset.

## Dependencies

Use the following command to install the required packages:

```bash
pip install -r requirements.txt
```

- `torch` and `torchvision` are required for building and training the CNN model.
- `matplotlib` is required for plotting the training and validation loss and accuracy.
- `numpy` and `pandas` are required for data manipulation.
- `torchsummary` is required for displaying the model summary.
- `tqdm` is required for displaying the progress bar.
- `scikit-learn` is required for feature analysis.

## File Structure

The project is organized as follows:

- `data/`: Contains the MNIST dataset.
- `features/`: Contains the visualization of extracted feature vectors.
- `models/`: Contains the implementation of the CNN architectures.
- `plots/`: Contains the plots of training and validation loss and accuracy.
- `results/`: Contains the comparison figures of the CNN models.
- `states/`: Contains the trained model weights.
- `main.py`: The main script for training and evaluating the CNN models.
- `feature_analysis.py`: The script for feature analysis.
- `plot.py`: The script for plotting comparison figures.
- `requirements.txt`: The file for installing the required packages.
- `test.sh` and `test.bat`: The test scripts for testing the CNN architectures and learning rate schedulers.
- `README.md`: The README file.

states and data folders are not packed due to size consideration.

## Usage

To train and evaluate the CNN models, run the following command:

```bash
python main.py
```

Configurable parameters include:

- `--model`: The CNN architecture to use. Default is `lenet`. Options include `lenet`, `alexnet`, `vgg11`, `googlenet`,
  `resnet18`, `squeezenet`, `shufflenetv2`, `mobilenetv3`, `mnasnet`, and `efficientnet`.
- `--batch_size`: The batch size for training. Default is `64`.
- `--epochs`: The number of epochs for training. Default is `50`. The default value is generous as early stopping is
  expected.
- `--lr`: The learning rate for training. Default is `1e-3`.
- `--dropout`: The dropout rate for training. Default is `0.5`.
- `--weight_decay`: The weight decay for training. Default is `1e-4`.
- `--patience`: The patience for early stopping. Default is `7`.
- `--lr_scheduler`: The learning rate scheduler. Default is `plateau`; options include `none`, `plateau`, and `cosine`.
- `--min_lr`: The minimum learning rate for both earning rate schedulers. Default is `1e-6`.
- `--factor`: The factor for the plateau learning rate scheduler. Default is `0.1`.
- `--lr_patience`: The patience for the plateau learning rate scheduler. Default is `3`.
- `--warmup_epochs`: The number of warmup epochs. Default is `5`.
- `--warmup_start_lr`: The initial learning rate for the warmup period. Default is `1e-6`.

To perform feature analysis, run the following command:

```bash
python feature_analysis.py
```

Configurable parameters include:

- `--model`: The CNN architecture to use, the same as above.
- `--batch_size`: The batch size for feature extraction. Default is `128`.
- `--n_components`: The number of components for t-SNE. Default is `2` for 2D visualization.
- `--perplexity`: The perplexity for t-SNE. Default is `30`.

## Scripts

Two test scripts are provided to test the combination of 10 CNN architectures and 2 learning rate schedulers.

The .bat version is for Windows, and the .sh version is for Linux. To run the script, simply execute the corresponding
script.

```bash
chmod +x test.sh
./test.sh
```

```bash
./test.bat
```

Please note that `python` is used for Windows while `python3` is used for Linux.

## Results

For accuracy, EfficientNet achieves the highest, but for time consideration, VGG11 is cost-effective.

For feature analysis, the feature vectors extracted from the CNN models are visualized in 2D using t-SNE. The feature
vectors are clustered into 10 groups, corresponding to the 10 classes of the MNIST dataset. The feature vectors are
colored according to the ground truth labels. The feature vectors are well separated, indicating that the CNN models are
effective in extracting discriminative features.

For the ARI of clustering, ShuffleNet and SqueezeNet have the highest for DBSCAN and K-means.

