import h5py

# Open the .h5 file
with h5py.File('/home/vatsal/FourcastNet/FourCastNet/output/scratch/directory/ \/autoregressive_predictions_tp.h5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())

    for key, value in f.items():
        print(f"{key}: {value[:][1]}")
    # print("Accuracy")
    # print(f['acc'])

    # print("Acc unweighted")
    # print(f['acc_unweighted'])

    # print("RMSE")
    # print(f['rmse'])

    # print("tqe")
    # print(f['tqe'])