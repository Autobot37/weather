import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
##2019, 5, 1]
##end_date: [2019, 7, 1]
def preprocess_sevir_dataset():
    sevir_data_dir = Path("/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/data/vil")
    output_base_dir = Path("/home/vatsal/NWM/CasCast/pixel_data/sevir")
    
    train_dir = output_base_dir / "train_2h"
    val_dir = output_base_dir / "valid_2h" 
    test_dir = output_base_dir / "test_2h"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    h5_files = [
        Path("/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/data/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5"), 
        Path("/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5")
    ]    
    train_files = []
    val_files = []
    test_files = []
    
    for h5_file_path in tqdm(h5_files, desc="Processing H5 files"):
        
        try:
            with h5py.File(h5_file_path, 'r') as h5_file:
                vil_data = h5_file['vil'][:]
                
                print(f"VIL data shape: {vil_data.shape}")
                
                if len(vil_data.shape) != 4:
                    print(f"Unexpected shape: {vil_data.shape}")
                    continue
                
                num_events, height, width, num_frames = vil_data.shape
                
                if num_frames != 49:
                    print(f"Expected 49 frames, got {num_frames}")
                    continue
                
                # Extract year from file path
                year = None
                for part in h5_file_path.parts:
                    if part in ['2017', '2018', '2019']:
                        year = part
                        break
                
                if year is None:
                    year = '2019'  # fallback
                
                # Process each event
                for event_idx in tqdm(range(num_events), desc=f"Processing events in {h5_file_path.name}"):
                    event_data = vil_data[event_idx]  # Shape: (384, 384, 49)
                    
                    # Transpose to (49, 384, 384) for temporal-first format
                    event_data = event_data.transpose(2, 0, 1)  # (49, 384, 384)
                    
                    # Split into 3 overlapping samples
                    samples = [
                        event_data[0:25],   # frames 0->24
                        event_data[12:37],  # frames 12->36  
                        event_data[24:49]   # frames 24->49
                    ]

            
                    selected_dir = test_dir
                    file_list = test_files

                    for sample_idx, sample_data in enumerate(samples):
                        # Add channel dimension: (25, 384, 384) -> (25, 384, 384, 1)
                        sample_data = np.expand_dims(sample_data, axis=-1)
                        
                        # Create filename
                        base_name = h5_file_path.stem
                        npy_filename = f"vil-{year}-{base_name}.h5-{event_idx}-{sample_idx}.npy"

                        output_path = selected_dir / npy_filename
                        np.save(output_path, sample_data.astype(np.float32))
                        
                        # Add to the same list as the directory
                        file_list.append(npy_filename)
                            
        except Exception as e:
            print(f"Error processing {h5_file_path}: {e}")
    
    # with open(output_base_dir / "train_list.txt", 'w') as f:
    #     for filename in sorted(train_files):
    #         f.write(filename + '\n')
    
    # with open(output_base_dir / "val_list.txt", 'w') as f:
    #     for filename in sorted(val_files):
    #         f.write(filename + '\n')
    
    with open(output_base_dir / "test_list.txt", 'w') as f:
        for filename in sorted(test_files):
            f.write(filename + '\n')
    
    print(f"\nPreprocessing complete!")
    # print(f"Train samples: {len(train_files)}")
    # print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    print(f"Output directory: {output_base_dir}")

if __name__ == "__main__":
    preprocess_sevir_dataset()
