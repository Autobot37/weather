import h5py

# Replace this with the path to your .h5 file
# file_path = '/home/vatsal/Supreme/Fourcastnet_outofsample/out_of_sample/2018.h5'
file_path = '/home/vatsal/NWM/FourcastNet/FourCastNet/output/scratch/directory/ \/autoregressive_predictions_tp.h5'


with h5py.File(file_path, 'r') as f:
    print("Top-level keys:")
    for key in f.keys():
        print(key)
    # List all groups and datasets
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    print("File structure:")
    f.visititems(print_structure)

    # Optional: Explore attributes
    print("\nFile-level attributes:")
    for key, val in f.attrs.items():
        print(f"  {key}: {val}")