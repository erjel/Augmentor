# Modified from https://github.com/aleju/imgaug/blob/0101108d4fed06bc5056c4a03e2bcb0216dac326/imgaug/__init__.py
# Copyright (c) 2015 aleju
# See ./LICENCE for licensing information

"""Imports for package imgaug."""
from __future__ import absolute_import

# this contains some deprecated classes/functions pointing to the new
# classes/functions, hence always place the other imports below this so that
# the deprecated stuff gets overwritten as much as possible
from .imgaug import *  # pylint: disable=redefined-builtin

from . import augmentables as augmentables
from .augmentables import *
from . import augmenters
from . import parameters
from . import dtypes

__version__ = '0.4.0'
