import sys

sys.path.insert(0, "lib/")
sys.path.insert(0, "model/emd_simple")
from network import Network as CrowdDetEMDSimple
from inference import get_data
sys.path.pop(0)
sys.path.insert(0, "model/emd_refine")
from network import Network as CrowdDetEMDRefine
sys.path.pop(0)
sys.path.pop(0)
