import h5py
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

def process_single_file_optimized_cpu(args):
    h5_file_path, train_dir, val_dir, test_dir = args
    
    try:
        h5_file_path = Path(h5_file_path)
        train_files_local = []
        val_files_local = []
        test_files_local = []
        
        with h5py.File(h5_file_path, 'r') as h5_file:
            year = None
            for part in h5_file_path.parts:
                if part in ['2017', '2018', '2019']:
                    year = part
                    break
            
            if year is None:
                return train_files_local, val_files_local, test_files_local
            
            month = h5_file_path.name[-11]
            
            if year != '2019':
                selected_dir = train_dir
                file_list = train_files_local
            else:
                if month in ['1', '2', '3', '4', '5']:
                    selected_dir = val_dir
                    file_list = val_files_local
                else:
                    selected_dir = test_dir
                    file_list = test_files_local

            vil_data = h5_file['vil'][:]
            
            if len(vil_data.shape) != 4:
                return train_files_local, val_files_local, test_files_local
            
            vil_data = vil_data.astype(np.float32)
            
            time_windows = [
                vil_data[:, :, :, :25],
                vil_data[:, :, :, 12:37],
                vil_data[:, :, :, 24:50]
            ]
            
            base_name = h5_file_path.stem
            
            for window_idx in tqdm(range(len(time_windows)), desc=f"Preparing windows for {h5_file_path.name}", leave=False):
                window_data = time_windows[window_idx]
                save_operations = []
                for event_idx in tqdm(range(window_data.shape[0]), desc=f"Preparing events in window {window_idx}", leave=False):
                    file_name = f"vil-{year}-{base_name}-{event_idx}-{window_idx}.npy"
                    output_path = selected_dir / file_name
                    save_operations.append((output_path, window_data[event_idx].copy(), file_name))
            
                for output_path, data, file_name in tqdm(save_operations, desc=f"Saving files for {h5_file_path.name}", leave=False):
                    np.save(output_path, data)
                    file_list.append(file_name)
                del save_operations
                del window_data
        
        return train_files_local, val_files_local, test_files_local
        
    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")
        return [], [], []

def preprocess_sevir_dataset_optimized():
    sevir_data_dir = Path("/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/data/vil")
    output_base_dir = Path("~/Dataserver/vil").expanduser()
    
    train_dir = output_base_dir / "train_2h"
    val_dir = output_base_dir / "valid_2h" 
    test_dir = output_base_dir / "test_2h"
    
    for dir_path in tqdm([train_dir, val_dir, test_dir], desc="Creating directories"):
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Finding H5 files...")
    h5_files = list(sevir_data_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} H5 files")
    
    train_files = []
    val_files = []
    test_files = []
    
    start_time = time.time()
    
    print("Using optimized CPU processing with 2 threads...")
    args_list = [(str(h5_file), train_dir, val_dir, test_dir) for h5_file in h5_files]
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(
            executor.map(process_single_file_optimized_cpu, args_list),
            total=len(args_list),
            desc="Processing files with 2 CPU threads"
        ))
        
        for train_files_local, val_files_local, test_files_local in tqdm(results, desc="Collecting results"):
            train_files.extend(train_files_local)
            val_files.extend(val_files_local)
            test_files.extend(test_files_local)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    print("Writing file lists...")
    file_operations = [
        (output_base_dir / "train_list.txt", sorted(train_files)),
        (output_base_dir / "val_list.txt", sorted(val_files)),
        (output_base_dir / "test_list.txt", sorted(test_files))
    ]
    
    for file_path, file_list in tqdm(file_operations, desc="Writing list files"):
        with open(file_path, 'w') as f:
            f.write('\n'.join(file_list) + '\n')
    
    print(f"\nPreprocessing complete!")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    print(f"Output directory: {output_base_dir}")
    print(f"Average time per file: {processing_time/len(h5_files):.2f} seconds")

if __name__ == "__main__":
    preprocess_sevir_dataset_optimized()