import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# make the framework and the example training scripts importable, and run from
# the project root so the relative MNIST data paths resolve
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
os.chdir(ROOT)
