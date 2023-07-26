from neuron import h,gui
import pandas as pd
import os
import logging

from ..model_utils import create_synapses

logger = logging.getLogger(__name__)

def create_cell(path=None):
    import neuron
    h.load_file("import3d.hoc")
    h.load_file("nrngui.hoc")

    if path is None:
        path = os.path.dirname(os.path.realpath(__file__)) +'/'

    neuron.load_mechanisms(path)
    h.chdir(path)

    # Load main cell template
    h.load_file("template.hoc")

    # Instantiate the cell from the template
    cell = h.cADpyr229_L23_PC_8ef1aa6602(0, 11.17)

    syn_df = create_synapses('rat')

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    
    return cell, syn_df
