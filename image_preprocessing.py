import cv2
import numpy as np
import os
import pydicom
from tqdm import tqdm


def read_dicom(dicom_path):
    """Enhanced DICOM reading with improved preprocessing"""
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    if len(image.shape) == 3:
        image = image[:, :, 0]
 
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    image = cv2.equalizeHist(image)
    
    return image


def enhanced_preprocessing(image):
    """Enhanced preprocessing combining multiple techniques"""
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    enhanced = cv2.addWeighted(denoised.astype(np.float64), 0.7, 
                              gradient_magnitude, 0.3, 0)
    
    return enhanced.astype(np.uint8)


def multi_threshold_segmentation(image):
    """Multi-threshold segmentation"""
    _, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    adaptive_binary = cv2.adaptiveThreshold(image, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    combined = cv2.bitwise_and(otsu_binary, adaptive_binary)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return cleaned


def process_image(image_path):
    """
    Simple processing pipeline that only does thresholding and saves debug images.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        None - Only saves debug images
    """
    if image_path.lower().endswith('.dcm'):
        gray = read_dicom(image_path)
    else:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read the image")
 
    preprocessed = enhanced_preprocessing(gray)
 
    binary = multi_threshold_segmentation(preprocessed)
 
    os.makedirs("preprocessed", exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(f"preprocessed/{base_filename}_binary.png", binary)
    cv2.imwrite(f"preprocessed/{base_filename}_preprocessed.png", preprocessed)
    cv2.imwrite(f"preprocessed/{base_filename}_original.png", gray)


def process_directory(input_dir="input-data/bone-scan1"):
    """Process all images in directory with simple thresholding pipeline"""
    os.makedirs("preprocessed", exist_ok=True)
 
    image_extensions = ('.dcm', '.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    processed_files = 0

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    all_files.sort()
    
    print(f"Starting processing of {len(all_files)} files...")
    print("=" * 60)

    for filename in tqdm(all_files, desc="Processing images", unit="file"):
        input_path = os.path.join(input_dir, filename)
        
        try:
            process_image(input_path)
            processed_files += 1
                
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")
            continue
 
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files found: {len(all_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed processing: {len(all_files) - processed_files}")
    
    print(f"\nOutput directory:")
    print(f"  - preprocessed/ - Thresholded binary images, preprocessed images, and original images")
    print("=" * 60)


def main():
    process_directory()


if __name__ == "__main__":
    main()