"""JX-3D — 3D organoid reconstruction from brightfield Z-stacks.

Pipeline shape (see `pipeline.run`):

    load  ->  focus profile  ->  per-slice 2D segmentation
          ->  Z linking      ->  shape-from-focus measurement
          ->  spheroid meshing + viewer

The design constraint that drives all of it: a 4x / NA 0.20 brightfield stack is
not an optical section. Depth comes from *where the rim is sharpest*, not from
where there is signal.
"""

__version__ = "1.0.0"

from .config import Acquisition, Params
from .stack import ZStack, load_stack

__all__ = ["Acquisition", "Params", "ZStack", "load_stack", "__version__"]
