# @title Imports

import dataclasses
import datetime
import math
from typing import Optional

from IPython.display import HTML
from IPython import display
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import ipywidgets as widgets
import jax
# from jax.extend.core import JaxprEqn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
import haiku as hk
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning
from plotting_fns import *
from other_fns import *


MODEL_PATH = ""  "gencast_params_GenCast 1p0deg Mini _2019.npz"
DATA_PATH = ""  "source-era5_date-2019-03-29_res-1.0_levels-13_steps-04.nc"
STATS_DIR = ""  "stats/"


# @title Load the model

with open(MODEL_PATH, "rb") as f:
  ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
state = {}

task_config = ckpt.task_config
sampler_config = ckpt.sampler_config

print(f"Sampler config : {sampler_config}")
noise_config = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

assert data_valid_for_model(DATA_PATH, MODEL_PATH)


# @title Load weather data

with open(DATA_PATH, "rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(DATA_PATH.removesuffix(".nc")).items()]))

example_batch



# @title Plot example data

plot_size = 7
variable = "geopotential"
level = 500
steps = example_batch.dims["time"]


data = {
    " ": scale(select(example_batch, variable, level, steps), robust=True),
}
fig_title = variable
if "level" in example_batch[variable].coords:
  fig_title += f" at {level} hPa"

plot_data(data, fig_title, plot_size, robust=True)

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# @title Load normalization data

with open(STATS_DIR +"gencast_stats_diffs_stddev_by_level.nc", "rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"gencast_stats_mean_by_level.nc", "rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"gencast_stats_stddev_by_level.nc", "rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"gencast_stats_min_by_level.nc", "rb") as f:
  min_by_level = xarray.load_dataset(f).compute()


#@title Build jitted functions, and possibly initialize random weights


def construct_wrapped_gencast():
  """Constructs and wraps the GenCast Predictor."""
  predictor = gencast.GenCast(
      sampler_config=sampler_config,
      task_config=task_config,
      denoiser_architecture_config=denoiser_architecture_config,
      noise_config=noise_config,
      noise_encoder_config=noise_encoder_config,
  )

  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level,
  )

  predictor = nan_cleaning.NaNCleaner(
      predictor=predictor,
      reintroduce_nans=True,
      fill_value=min_by_level,
      var_to_clean='sea_surface_temperature',
  )

  return predictor

@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
  predictor = construct_wrapped_gencast()
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
  predictor = construct_wrapped_gencast()
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics),
  )


def grads_fn(params, state, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), i, t, f
    )
    return loss, (diagnostics, next_state)

  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True
  )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads




if params is None:
  init_jitted = jax.jit(loss_fn.init)
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets=train_targets,
      forcings=train_forcings,
  )


loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)
grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
# We also produce a pmapped version for running in parallel.
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")


# Run the model

# The number of ensemble members should be a multiple of the number of devices.
print(f"Number of local devices {len(jax.local_devices())}")


print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

num_ensemble_members = 1 # @param int
rng = jax.random.PRNGKey(0)


rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

chunks = []
for chunk in rollout.chunked_prediction_generator_multiple_runs(
    # Use pmapped version to parallelise across devices.
    predictor_fn=run_forward_pmap,
    rngs=rngs,
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
    num_steps_per_chunk = 1,
    num_samples = num_ensemble_members,
    pmap_devices=jax.local_devices()
    ):
    chunks.append(chunk)
    
predictions = xarray.combine_by_coords(chunks)

