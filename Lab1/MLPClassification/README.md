# Project 1: MLP-based Text Classification
### Dependencies
Use
```shell
pip install -r requirements.txt
```
to install the dependencies, including
- `matplotlib`
- `nltk`
- `numpy`
- `pandas`
- `scikit_learn`
- `seaborn`
- `torch`
- `tqdm`
- `transformers`


### Structure
- `BERT.py` contains the BERT-based model.
- `TF-IDF.py` contains the TFIDF-based model.'
- `train_data.txt` and `test.txt` are the training and testing data.
- `results.txt` is the final chosen output of the test data.
- `mlp_epoch_stats.json` and `mlp_analysis.json` are data for analysis.
- `requirements.txt` contains the dependencies.
- `README.md` is this file.

### Usage for BERT
Run
```shell
python BERT.py
```
Configurable parameters include
- `train_file`: default to `train_data.txt`
- `test_file`: default to `test.txt`
- `output_file`: default to `result`
- `batch_size`: default to 32
- `epochs`: default to 10
- `learning_rate`: default to 2e-5
- `weight_decay`: default to 0.01
- `dropout_rate`: default to 0.5
- `lr_patience`: default to 2, the patience for learning rate reduction
- `lr_factor`: default to 0.1, the factor by which to reduce learning rate

Example:
```shell
python BERT.py --batch_size 64 --epochs 5
```

Alternatively, in the working directory, run
```shell
./execute_bert.bat
```
or
```shell
./execute_tfidf.bat
```
to run the BERT-based model or the TFIDF-based model with default parameters.

### Analysis
Two json files are stored:
- `mlp_epoch_stats.json` contains the training and validation loss and accuracy of each epoch.
- `mlp_analysis.json` contains the best result for each layer count.
Three hidden layer configuration are tested, no hidden layer, one 512 hidden layer and two 512-256 hidden layers.

The figures are stored in the directory.

### Results
In the validation set, BERT-based model achieves an accuracy of 0.95 and is chosen as the final model.  