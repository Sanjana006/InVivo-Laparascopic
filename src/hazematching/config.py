"""
src/hazematching/config.py
==========================
Registers the 'laparoscopy' dataset subset within HazeMatching.
"""

from hazematching.datasets import config as hm_config

# The name of our custom dataset
LAPAROSCOPY_SUBSET = "laparoscopy"

def register_laparoscopy_subset():
    """Monkey-patch the HazeMatching config to recognize our new subset."""
    if LAPAROSCOPY_SUBSET not in hm_config.SUBSETS:
        # Add to SUBSETS
        hm_config.SUBSETS = tuple(list(hm_config.SUBSETS) + [LAPAROSCOPY_SUBSET])
        
        # Add to DATASET_DESCRIPTIONS
        hm_config.DATASET_DESCRIPTIONS[LAPAROSCOPY_SUBSET] = "InVivo Laparoscopy"
        
        # Update ALIASES mapping
        hm_config._ALIASES[hm_config._normalise_name(LAPAROSCOPY_SUBSET)] = LAPAROSCOPY_SUBSET
        hm_config._ALIASES[hm_config._normalise_name("InVivo Laparoscopy")] = LAPAROSCOPY_SUBSET

        try:
            from .norm_stats import inject_stats
            inject_stats()
        except ImportError:
            pass # Stats not generated yet

# Execute the registration when this module is imported
register_laparoscopy_subset()
