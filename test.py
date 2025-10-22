import flask
import tensorflow as tf
import importlib.metadata

print("Flask version:", importlib.metadata.version("flask"))
print("TensorFlow version:", tf.__version__)
