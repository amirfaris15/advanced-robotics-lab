"""Shared geometric constants learned by the inverse-geometry solver."""

import numpy as np
import pinocchio as pin


# The value is updated whenever a valid bimanual grasp is found.  It is useful
# for debugging: while carrying a rigid cube, this hand-to-hand transform is
# expected to remain constant.
LMRREF = pin.SE3.Identity()
RMLREF = pin.SE3.Identity()


def remember_hand_relation(left_hand_placement, right_hand_placement):
    """Store and return the right hand expressed in the left-hand frame."""
    global LMRREF, RMLREF
    LMRREF = left_hand_placement.inverse() * right_hand_placement
    RMLREF = LMRREF.inverse()
    return LMRREF
