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
        path = os.path.dirname(os.path.realpath(__file__))

    model =  "cell0603_11_model_937"
    model_path = path + '/'  + model
    morph = path +'/2013_03_06_cell11_1125_H41_06.ASC'
    dend_secs = ['dend','apic']

    neuron.load_mechanisms(path)

    # load cell
    h.load_file( model_path+".hoc")
    cell_template = getattr(h,model)

    cell = cell_template()
    nl = h.Import3d_Neurolucida3()

    # Creating the model
    nl.quiet = 1
    nl.input(morph)
    imprt = h.Import3d_GUI(nl, 0)   
    imprt.instantiate(cell)    
    cell.indexSections(imprt)
    cell.geom_nsec()   
    cell.geom_nseg()
    cell.delete_axon()
    cell.insertChannel()
    cell.init_biophys()
    cell.biophys()

    syn_df = create_synapses('human')

    # explicitly overriding default temperature
    celsius = 37.0
    logger.info(f"Setting temperature to be {celsius} degree celsius")
    h.celsius = celsius

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    return cell, syn_df