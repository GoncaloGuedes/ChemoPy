""" Data loader module for chemopy. """

from ._perkin_elmer import load_perkin_elmer_data
from ._texas_instruments import load_texas_instruments_data

__all__ = ["load_perkin_elmer_data", "load_texas_instruments_data"]
