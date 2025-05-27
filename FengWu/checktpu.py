import jax

devices = jax.devices()
print(devices)
if any(device.platform == 'tpu' for device in jax.devices()):
    print("TPU is available!")
else:
    print("TPU not found.")
