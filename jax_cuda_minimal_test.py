# A minimal test file to check whether the installed JAX packages can "see" the CUDA hardware
# and whether JAX can allocate the GPU memory using the right ptxas file

import jax 
import jax.numpy as jnp

dev = jax.devices()

if dev =! 'gpu':
  print("Your code is not running on GPU")
else:
  print("Congrats! JAX runs on CUDA device")

# create a random test array
x = jnp.ones(100,100)


