from shutil import copyfile
from prediff.datasets.sevir.sevir_dataloader import (
    SEVIRDataLoader, SEVIR_LR_DATA_DIR,
    SEVIR_CATALOG, SEVIR_LR_CATALOG,)

print("SEVIR low resolution data will be saved at:", SEVIR_LR_DATA_DIR)
print("SEVIR low resolution catalog will be saved at:", SEVIR_LR_CATALOG)
print(SEVIR_CATALOG)
print()
if __name__ == '__main__':
    downsample_dict = {'vil': (2, 3, 3)}
    batch_size = 32
    SEVIR_CATALOG = "/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/CATALOG.csv"
    copyfile(SEVIR_CATALOG, SEVIR_LR_CATALOG)
    sevir_dataloader = SEVIRDataLoader(data_types=['vil', ], sample_mode='sequent', batch_size=batch_size, 
                                       sevir_catalog=SEVIR_CATALOG, sevir_data_dir="/home/vatsal/NWM/earth-forecasting-transformer/datasets/sevir/data")
    sevir_dataloader.save_downsampled_dataset(
        save_dir=SEVIR_LR_DATA_DIR,
        downsample_dict=downsample_dict,
        verbose=True)
