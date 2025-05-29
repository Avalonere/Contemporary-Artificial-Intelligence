# AI Project 4: Seq2Seq Text Generation

## Dependencies
To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Overview
The project is divided into two parts - RNN and Transformer based models, each in the respective folder. We implemented five models
- `RNN`
- `LSTM`
- `GRU`
- `BART`
- `T5`
to generate diagnosis from description in the given dataset.

For the parameters, `config.py` is provided. For the transformer models, it can also be set with `argparse` parameters,
including the model to use (`bart` or `t5`), batch size, epochs, learning rate and epochs.

An experiment to test `bart` and `t5` in different seeds is implemented in the script. To run the experiment, use the following command:
```bash
chmod +x experiment.sh
./experiment.sh
```
Alternatively, on Windows,
```powershell
./experiment.bat
```

## Results
The models are evaluated with `BLEU`, `ROUGE` and `METEOR` scores. Generally, Transformer based models outperform RNN based ones with a considerable margin, achieving a satisfactory `BLEU` score. For the details, please refer to the report.