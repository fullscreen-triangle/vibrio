# Vibrio: Project Setup Guide

This document outlines the complete setup process for the Vibrio Human Speed Analysis Framework.

## Project Folder Structure

```
vibrio/
├── main.py                    # Main entry point
├── calibrate.py               # Camera calibration utility
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License file
├── setup.md                   # This setup guide
├── modules/                   # Core modules directory
│   ├── __init__.py           # Package initialization
│   ├── detector.py           # Human detection module
│   ├── tracker.py            # Human tracking module
│   ├── speed_estimator.py    # Speed calculation module
│   ├── physics_verifier.py   # Physics-based verification
│   ├── visualizer.py         # Visualization and output
│   └── utils.py              # Utility functions
├── models/                    # Pre-trained models directory
│   └── yolov8n.pt            # YOLOv8 nano model (downloaded automatically)
├── data/                      # Data directory
│   ├── calibration/          # Camera calibration files
│   │   └── example.json      # Example calibration file
│   └── test_videos/          # Test video files
├── results/                   # Results output directory
│   ├── videos/               # Processed videos
│   ├── plots/                # Generated plots
│   └── data/                 # Extracted data files
└── web/                       # Web interface (future extension)
    ├── app.py                # Flask/FastAPI application
    ├── static/               # Static assets
    └── templates/            # HTML templates
```

## Environment Setup

### System Requirements

- Python 3.8+ (Python 3.10 recommended)
- pip and virtualenv or conda
- Git
- OpenCV dependencies (system libraries)
- CUDA toolkit (optional, for GPU acceleration)

### Python Dependencies

The `requirements.txt` file contains all necessary Python packages:

```
opencv-python>=4.5.0           # Computer vision library
numpy>=1.20.0                  # Numerical computing
torch>=1.9.0                   # Deep learning framework
torchvision>=0.10.0            # Computer vision for PyTorch
ultralytics>=8.0.0             # YOLOv8 implementation
filterpy>=1.4.5                # Kalman filtering library
scipy>=1.7.0                   # Scientific computing
matplotlib>=3.4.0              # Plotting and visualization
tqdm>=4.62.0                   # Progress bars
```

## Installation Methods

You have two options for installing the Vibrio framework:

### Method 1: Development Installation (Recommended for Development)

For development work where you'll be modifying the source code:

```bash
# Clone the repository
git clone https://github.com/yourusername/vibrio.git
cd vibrio

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

This will install the package in "editable" mode, meaning changes to the source code will be immediately reflected without needing to reinstall.

### Method 2: Direct Installation

For users who just want to use the package:

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/vibrio.git

# Or, after downloading the source:
cd vibrio
pip install .
```

### Package Structure for Installation

When using `setup.py`, the project structure should follow this convention:

```
vibrio/
├── setup.py                   # Package installation script
├── README.md                  # Documentation
├── LICENSE                    # License file
├── requirements.txt           # Dependencies list
├── vibrio/                    # Package directory (note the nesting)
│   ├── __init__.py            # Makes vibrio a package
│   ├── main.py                # Main module
│   ├── calibrate.py           # Calibration module
│   ├── modules/               # Sub-package
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   └── ...
```

After installation, you can use the package from any directory by importing it:

```python
import vibrio
```

Or use the command-line tools defined in the entry_points:

```bash
# Run the main application
vibrio --input your_video.mp4 --output results/

# Run the calibration tool
vibrio-calibrate --input calibration_video.mp4 --output calibration.json
```

## Step-by-Step Setup Instructions

### 1. Clone the Repository (if applicable)

```bash
git clone https://github.com/yourusername/vibrio.git
cd vibrio
```

### 2. Create and Activate Virtual Environment

#### Using venv (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

#### Using conda

```bash
# Create conda environment
conda create -n vibrio python=3.10
conda activate vibrio
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Required Directories

```bash
mkdir -p modules models data/calibration data/test_videos results/videos results/plots results/data
```

### 5. Install CUDA (Optional, for GPU Support)

For GPU acceleration with NVIDIA GPUs, install the appropriate CUDA toolkit and cuDNN library. Follow the installation guides at:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

Ensure your PyTorch installation is CUDA-compatible:

```bash
# Check if PyTorch recognizes GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 6. Generate Example Calibration File

```bash
mkdir -p data/calibration
python -c "from modules.utils import create_calibration_template; create_calibration_template('data/calibration/example.json')"
```

## Development Setup

### IDE Configuration

#### Visual Studio Code

Recommended extensions:
- Python
- Pylance
- autoDocstring
- GitLens
- Jupyter

Settings:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

#### PyCharm

Recommended configuration:
- Enable PEP 8 compliance checking
- Configure virtual environment as project interpreter
- Enable Scientific mode for data visualization

### Development Workflow

1. Create new modules in the `modules/` directory
2. Run tests with pytest (to be implemented)
3. Use the main application with:
   ```bash
   python main.py --input your_video.mp4 --output results/
   ```
4. For camera calibration:
   ```bash
   python calibrate.py --input calibration_video.mp4 --output data/calibration/your_calibration.json --show
   ```

## Common Issues and Solutions

### Missing System Libraries for OpenCV

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

#### On macOS:
```bash
brew install cmake pkg-config
```

### YOLOv8 Model Download Issues

If the automatic YOLOv8 model download fails:
1. Manually download the model from [Ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
2. Place it in the `models/` directory

### CUDA Compatibility Issues

Ensure torch is installed with the correct CUDA version:
```bash
# For CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

## Deployment Considerations

### Docker Support (Planned)

A Dockerfile will be provided in the future for containerized deployment.

### Production Environment

For production use:
1. Increase logging levels
2. Consider using a WSGI server for web interface
3. Implement proper error handling and recovery
4. Use a queue system for processing multiple videos

## Future Development

- React.js frontend for web interface
- API endpoints for third-party integration
- Database storage for analysis results
- Cloud deployment options
- Mobile application integration

## Contact and Support

For issues or questions, refer to the project README or create issues in the repository.
