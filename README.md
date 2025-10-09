# Bone Pore Simulator

A comprehensive tool for processing DICOM bone scan images and simulating porous structures for biomedical research applications.

## 🔬 Overview

This project provides a complete pipeline for:
- Processing DICOM bone scan images with advanced segmentation techniques
- Generating realistic pore structures within bone tissue
- Creating visualizations for research and analysis
- Supporting both 2D pore simulation and preparation for 3D analysis

## 📋 Features

### Image Processing Pipeline
- **DICOM Support**: Direct reading and processing of medical DICOM files
- **Advanced Preprocessing**: Bilateral filtering, gradient enhancement, and histogram equalization
- **Dual Thresholding**: Combined Otsu and adaptive thresholding for robust bone segmentation
- **Morphological Cleaning**: Automated noise removal and structure refinement

### Pore Simulation
- **Realistic Pore Generation**: 3-7 complete circular pores per bone slice
- **Smart Placement**: Distance transform ensures pores are fully contained within bone regions
- **Non-overlapping**: Automatic spacing to prevent pore intersection
- **Customizable Parameters**: Adjustable pore size, count, and placement criteria

### Visualization
- **Overlay Generation**: Red pore highlighting on original bone images
- **Multi-format Output**: Binary masks, processed images, and overlay visualizations

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd acc-proj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `opencv-python`: Image processing and computer vision
- `pydicom`: DICOM medical image handling
- `porespy`: Porous media analysis toolkit
- `scipy`: Scientific computing and morphological operations
- `numpy`: Numerical computing
- `tqdm`: Progress bar visualization

## 📁 Project Structure

```
acc-proj/
├── image_preprocessing.py      # Main DICOM processing pipeline
├── 2Dpore_simulation.py       # Pore generation and simulation
├── requirements.txt           # Python dependencies
├── input-data/
│   └── bone-scan1/           # DICOM files (168 slices)
├── preprocessed/             # Processed images output
│   ├── *_original.png        # Normalized DICOM images
│   ├── *_preprocessed.png    # Enhanced images
│   └── *_binary.png          # Binary bone masks
└── simulated/                # Pore simulation output
    ├── *_porous.png          # Binary images with pores
    ├── *_overlay.png         # Visualization overlays
    └── *_original.png        # Reference images
```

## 🔧 Usage

### 1. Image Preprocessing
Process DICOM files to create clean bone segmentation masks:

```bash
python image_preprocessing.py
```

**Output**: Creates three types of images in `preprocessed/` directory:
- `*_original.png`: Normalized DICOM images
- `*_preprocessed.png`: Edge-enhanced, denoised images  
- `*_binary.png`: Binary bone masks (white=bone, black=background)

### 2. Pore Simulation
Generate realistic pores within bone structures:

```bash
python 2Dpore_simulation.py
```

**Output**: Creates pore simulations in `simulated/` directory:
- `*_porous.png`: Binary masks with simulated pores
- `*_overlay.png`: Original images with red pore overlays
- `*_original.png`: Reference normalized images

## 🔬 Technical Details

### Image Processing Pipeline

1. **DICOM Loading**: 
   - Extracts pixel arrays from medical DICOM files
   - Normalizes 16-bit medical data to 8-bit range (0-255)
   - Applies histogram equalization for contrast enhancement

2. **Enhanced Preprocessing**:
   - Bilateral filtering for edge-preserving noise reduction
   - Sobel gradient calculation for edge detection
   - Weighted combination (70% original + 30% gradient information)

3. **Multi-threshold Segmentation**:
   - Otsu thresholding for global optimal threshold
   - Adaptive thresholding for local variations
   - Logical AND combination for conservative segmentation
   - Morphological operations (closing + opening) for cleanup

### Pore Generation Algorithm

1. **Distance Transform**: Calculates distance from bone edges
2. **Valid Location Detection**: Ensures pores fit completely within bone
3. **Random Placement**: Generates 3-7 pores with random sizes (5-12 pixel radius)
4. **Overlap Prevention**: Updates distance map after each pore placement
5. **Quality Assurance**: Verifies complete containment within bone regions

## 📝 File Formats

### Input
- **DICOM (.dcm)**: Medical imaging standard files
- **Standard Images**: PNG, JPG, JPEG, BMP, TIFF support

### Output
- **PNG Images**: All processed and simulated results
- **Binary Masks**: 8-bit grayscale (0=background, 255=bone/pores)
- **RGB Overlays**: Color-coded visualizations

*Built for biomedical research and porous media analysis at Purdue University*