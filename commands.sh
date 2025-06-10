 export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#  source SSBware/senv/bin/activate
#   source senv/bin/activate

MASTER_ADDR=localhost MASTER_PORT=10001 python3 /home/vatsal/NWM/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 1 --cfg /home/vatsal/NWM/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml --save tmp_sevir3
# python -m pip install gdown
