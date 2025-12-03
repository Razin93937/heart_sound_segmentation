# Heart Sound Segmentation

This project implements a heart sound segmentation pipeline using Temporal Convolutional Networks (TCNs) with Bidirectional GRU and a small Flask web demo.

# Video Demonstration
[![Watch the video](https://img.youtube.com/vi/STgNDhXaLs4/0.jpg)](https://youtu.be/STgNDhXaLs4)

## Setup

Create virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Training

Train a model using the Circor Digiscope Phonocardiogram dataset:
```bash
python model/train.py --data_dir /path/to/dataset --epochs 50 --batch_size 32
```

Optional flags: `--window_size 4000`, `--base_channels 256`, `--learning_rate 1e-3`, `--mixed_precision`

## Inference

Run inference on a single WAV file or directory:
```bash
python model/inference.py --model_path model/best_models/your_model.keras --wav /path/to/file.wav
```

With visualization:
```bash
python model/inference.py --model_path model/best_models/your_model.keras --wav /path/to/file.wav --plot --plot_outdir output/
```

## Flask Web GUI

Start the web application:
```bash
python webapp/app.py
```

Access at `http://localhost:5555`

## Additional Information
- Developed as a term project for BME 310 "Communicating Protocols for Biomedical Instruments Sessional".
- Training designed for the Circor Digiscope Phonocardiogram dataset 1.0.3.
