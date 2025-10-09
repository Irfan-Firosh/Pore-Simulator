import cv2
import numpy as np
import os
import porespy as ps
import matplotlib.pyplot as plt
from skimage import morphology
import pydicom
from tqdm import tqdm


def read_dicom(dicom_path):
    """Read DICOM file and normalize"""
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    image = cv2.equalizeHist(image)
    
    return image


def generate_pores_in_bone(binary_mask, num_pores=None, min_pore_size=5, max_pore_size=12):
    """
    Generate 3-7 complete circular pores within the white bone regions
    
    Args:
        binary_mask: Binary image where white = bone, black = background
        num_pores: Number of pores to generate (3-7), if None will be random
        min_pore_size: Minimum pore radius in pixels
        max_pore_size: Maximum pore radius in pixels
    
    Returns:
        porous_structure: Binary image with pores added to bone regions
    """
    bone_mask = binary_mask.copy()
    
    bone_area = np.sum(bone_mask > 0)
    if bone_area == 0:
        return bone_mask
    
    if num_pores is None:
        num_pores = np.random.randint(3, 8)  # 3 to 7 pores
    
    porous_bone = bone_mask.copy()
    
    from scipy import ndimage
    bone_regions = bone_mask > 0
    distance_map = ndimage.distance_transform_edt(bone_regions)
    
    pores_created = 0
    attempts = 0
    max_attempts = 1000
    
    while pores_created < num_pores and attempts < max_attempts:
        radius = np.random.uniform(min_pore_size, max_pore_size)
        
        valid_locations = distance_map >= (radius + 2)  # +2 for safety margin
        
        if not np.any(valid_locations):
            radius = min_pore_size
            valid_locations = distance_map >= (radius + 1)

        if np.any(valid_locations):
            valid_coords = np.where(valid_locations)
            
            idx = np.random.randint(0, len(valid_coords[0]))
            center_y, center_x = valid_coords[0][idx], valid_coords[1][idx]
            
            y, x = np.ogrid[:bone_mask.shape[0], :bone_mask.shape[1]]
            pore_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            if np.all(bone_regions[pore_mask]):
                porous_bone[pore_mask] = 0
                
                distance_map[pore_mask] = 0
                distance_map = ndimage.distance_transform_edt(porous_bone > 0)
                
                pores_created += 1
        
        attempts += 1
    
    return porous_bone


def generate_realistic_pores(binary_mask, num_pores=None):
    """
    Generate 3-7 complete pores using the improved method
    """
    return generate_pores_in_bone(binary_mask, num_pores)


def create_overlay_visualization(original_image, porous_structure, binary_mask):
    """
    Create visualization overlaying pores on original image (without green contours)
    """
    if len(original_image.shape) == 2:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = original_image.copy()
    
    bone_regions = binary_mask > 0
    pore_locations = bone_regions & (porous_structure == 0)
    
    overlay[pore_locations] = [255, 0, 0]  # Red pores only
    
    return overlay


def process_image_with_pores(image_path, num_pores=None):
    """
    Process a single image to add 3-7 simulated pores
    """
    if image_path.lower().endswith('.dcm'):
        original = read_dicom(image_path)
    else:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError("Could not read the image")
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    binary_path = f"preprocessed/{base_filename}_binary.png"
    
    if not os.path.exists(binary_path):
        print(f"Warning: Binary mask not found for {base_filename}")
        return None
    
    binary_mask = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    if num_pores is None:
        num_pores = np.random.randint(3, 8)
    porous_structure = generate_realistic_pores(binary_mask, num_pores)
    
    overlay = create_overlay_visualization(original, porous_structure, binary_mask)
    
    os.makedirs("simulated", exist_ok=True)
    
    cv2.imwrite(f"simulated/{base_filename}_porous.png", porous_structure)
    
    cv2.imwrite(f"simulated/{base_filename}_overlay.png", overlay)
    
    cv2.imwrite(f"simulated/{base_filename}_original.png", original)
    
    return porous_structure, overlay


def process_directory(input_dir="input-data/bone-scan1"):
    """
    Process all images in directory to generate 3-7 complete pores per image
    """
    os.makedirs("simulated", exist_ok=True)
    
    image_extensions = ('.dcm', '.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    processed_files = 0
    
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    all_files.sort()
    
    print(f"Starting pore simulation for {len(all_files)} files...")
    print("Generating 3-7 complete pores per image")
    print("=" * 60)
    
    for filename in tqdm(all_files, desc="Simulating pores", unit="file"):
        input_path = os.path.join(input_dir, filename)
        
        try:
            result = process_image_with_pores(input_path)
            
            if result is not None:
                processed_files += 1
            else:
                tqdm.write(f"Skipped {filename} (no binary mask found)")
                
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("PORE SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total files found: {len(all_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed processing: {len(all_files) - processed_files}")
    print(f"Pores per image: 3-7 complete circular pores")
    
    print(f"\nOutput directory:")
    print(f"  - simulated/ - Contains:")
    print(f"    - *_porous.png - Binary images with simulated pores")
    print(f"    - *_overlay.png - Original images with red pore overlay")
    print(f"    - *_original.png - Original normalized images")
    print("=" * 60)


def main():
    process_directory()


if __name__ == "__main__":
    main()
