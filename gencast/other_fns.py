# @title Check example dataset matches model
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


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_")[:3])

def data_valid_for_model(file_name: str, params_file_name: str):
  """Check data type and resolution matches."""
  data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  res_matches = data_file_parts["res"].replace(".", "p") in params_file_name.lower()
  source_matches = "Operational" in params_file_name
  if data_file_parts["source"] == "era5":
    source_matches = not source_matches
  return res_matches and source_matches

