import os

# Never let unit tests see a GPU, irrespective of availability on this shared host.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
