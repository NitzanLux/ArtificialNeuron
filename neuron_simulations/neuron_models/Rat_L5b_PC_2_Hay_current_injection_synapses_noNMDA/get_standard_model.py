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

    morphologyFilename = path + "/morphologies/cell1.asc"
    biophysicalModelFilename = path + "/L5PCbiophys5b.hoc"
    biophysicalModelTemplateFilename = path + "/L5PCtemplate_2.hoc"

    h.load_file(biophysicalModelFilename)
    h.load_file(biophysicalModelTemplateFilename)
    cell = h.L5PCtemplate(morphologyFilename, 24.30) # to get 1041 segs

    syn_df = create_synapses('rat_current_injection_synapse_noNMDA')

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    
    return cell, syn_df
