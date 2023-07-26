from neuron import h,gui
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

PARAMETER_SETS = {
    'human':{
        'AMPANMDA_e': 0,
        'tau_r_AMPA':0.3,
        'tau_d_AMPA':1.8,
        'tau_r_NMDA':8.019,
        'tau_d_NMDA':34.9884,
        'gamma':0.0765685,
        'NMDA_ratio':0.00131038/0.00073027,
        'AMPA_default_conductance':0.00073027,
        
        'GABAA_e': -80,
        'tau_r_GABAA':0.2,
        'tau_d_GABAA':8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance':0.0007,

        'celsius':34.0,
    },

    'rat':{
        'AMPANMDA_e': 0,
        'tau_r_AMPA':0.2,
        'tau_d_AMPA':1.7,
        'tau_r_NMDA':8.019,
        'tau_d_NMDA':34.9884,
        'gamma':0.0765685,
        'NMDA_ratio':0.0005/0.0007,
        'AMPA_default_conductance':0.0007,
        
        'GABAA_e': -80,
        'tau_r_GABAA':0.2,
        'tau_d_GABAA':8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance':0.0007,

        'celsius':34.0,
    }
    ,
    'rat_current_injection_synapse': {
        'AMPANMDA_e': 0,
        'tau_r_AMPA': 0.2,
        'tau_d_AMPA': 1.7,
        'tau_r_NMDA': 8.019,
        'tau_d_NMDA': 34.9884,
        'gamma': 0.0765685,
        'NMDA_ratio': 0.0005 / 0.0007,
        'AMPA_default_conductance': 0.0007,
        'e_pas':-70.,
        'GABAA_e': -80,
        'tau_r_GABAA': 0.2,
        'tau_d_GABAA': 8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance': 0.0007,

        'celsius': 34.0,
    }
    ,
    'rat_current_injection_synapse_noNMDA': {
        'AMPANMDA_e': 0,
        'tau_r_AMPA': 0.2,
        'tau_d_AMPA': 1.7,
        'tau_r_NMDA': 8.019,
        'tau_d_NMDA': 34.9884,
        'gamma': 0.0765685,
        'NMDA_ratio': 0/ 0.0007,
        'AMPA_default_conductance': 0.0007,
        'e_pas': -70.,
        'GABAA_e': -80,
        'tau_r_GABAA': 0.2,
        'tau_d_GABAA': 8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance': 0.0007,

        'celsius': 34.0,
    }
    ,

    'rat_david':{
        'AMPANMDA_e': 0,
        'tau_r_AMPA':0.3,
        'tau_d_AMPA':3,
        'tau_r_NMDA':2,
        'tau_d_NMDA':70,
        'gamma':0.08,
        'NMDA_ratio':0.0004/0.0004,
        'AMPA_default_conductance':0.0004,
        
        'GABAA_e': -80,
        'tau_r_GABAA':0.2,
        'tau_d_GABAA':8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance':0.0010,

        'celsius':6.3,
    },

    'rat_noNMDA':{
        'AMPANMDA_e': 0,
        'tau_r_AMPA':0.2,
        'tau_d_AMPA':1.7,
        'tau_r_NMDA':8.019,
        'tau_d_NMDA':34.9884,
        'gamma':0.0765685,
        'NMDA_ratio':0/0.0007,
        'AMPA_default_conductance':0.0007,
        
        'GABAA_e': -80,
        'tau_r_GABAA':0.2,
        'tau_d_GABAA':8,
        'GABAB_ratio': 0,
        'GABAA_default_conductance':0.0007,

        'celsius':34.0,
    }
}

def create_synapses(parameter_set_name):
    params = PARAMETER_SETS[parameter_set_name]

    logger.info(f'Creating synapses for parameter set: {parameter_set_name}')

    dend_secs = ['dend','apic']
    num_segments = 0
    all_segments = []
    seg_lens  = []
    for sec in h.allsec(): 
        if sum([1 for i in dend_secs if i in sec.name()]):
            num_segments+= sec.nseg
            for seg in sec:
                all_segments.append(seg)
                seg_lens.append(seg.sec.L/sec.nseg)


    # Create excitatory and inhibitory synapses per segment
    exc_synapses = []
    exc_netcons = []
    inh_synapses = []
    inh_netcons =[]
    for seg in all_segments:
        if 'old_impl' in params and params['old_impl']:
            AMPANMDA = h.ProbAMPANMDA2(seg)
            if parameter_set_name =='rat_current_injection_synapse':
                AMPANMDA.e_pas = params['e_pas']
            AMPANMDA.tau_r_AMPA = params['tau_r_AMPA']
            AMPANMDA.tau_d_AMPA = params['tau_d_AMPA']
            AMPANMDA.tau_r_NMDA = params['tau_r_NMDA']
            AMPANMDA.tau_d_NMDA = params['tau_d_NMDA']
            if 'old_weight' in params and params['old_weight']:
                AMPANMDA.gmax = params['AMPA_default_conductance']
            else:
                AMPANMDA.gmax = 1
            AMPANMDA.e = params['AMPANMDA_e']
            AMPANMDA.Use = 1
            AMPANMDA.u0 = 0
            AMPANMDA.Dep = 0
            AMPANMDA.Fac = 0
            AMPANMDA_ncon = h.NetCon(None, AMPANMDA)
            if 'old_weight' in params and params['old_weight']:
                AMPANMDA_ncon.weight[0] = 1
            else:
                AMPANMDA_ncon.weight[0] = params['AMPA_default_conductance']
        else:
            AMPANMDA = h.AMPANMDA_EMS(seg)
            if parameter_set_name == 'rat_current_injection_synapse':
                AMPANMDA.e_pas = params['e_pas']
            AMPANMDA.e = params['AMPANMDA_e']
            AMPANMDA.tau_r_AMPA = params['tau_r_AMPA']
            AMPANMDA.tau_d_AMPA = params['tau_d_AMPA']
            AMPANMDA.tau_r_NMDA = params['tau_r_NMDA']
            AMPANMDA.tau_d_NMDA = params['tau_d_NMDA']
            AMPANMDA.gamma = params['gamma']
            AMPANMDA.NMDA_ratio = params['NMDA_ratio']
            AMPANMDA_ncon = h.NetCon(None, AMPANMDA)
            AMPANMDA_ncon.weight[0] = params['AMPA_default_conductance']

        exc_synapses.append(AMPANMDA)
        exc_netcons.append(AMPANMDA_ncon)

    # for naming, it is better to run it twice
    for seg in all_segments:
        if 'old_impl' in params and params['old_impl']:
            GABAAB = h.ProbUDFsyn2_current(seg)
            if parameter_set_name == 'rat_current_injection_synapse':
                GABAAB.e_pas = params['e_pas']
            GABAAB.tau_r = params['tau_r_GABAA']
            GABAAB.tau_d = params['tau_d_GABAA']
            GABAAB.e = params['GABAA_e']
            if 'old_weight' in params and params['old_weight']:
                GABAAB.gmax = params['GABAA_default_conductance']
            else:
                GABAAB.gmax = 1
            GABAAB.Use = 1
            GABAAB.u0 = 0
            GABAAB.Dep = 0
            GABAAB.Fac = 0
            GABAAB_ncon = h.NetCon(None, GABAAB)
            if 'old_weight' in params and params['old_weight']:
                GABAAB_ncon.weight[0] = 1
            else:
                GABAAB_ncon.weight[0] = params['GABAA_default_conductance']
        else:
            GABAAB = h.GABAAB_EMS(seg)

            if parameter_set_name == 'rat_current_injection_synapse':
                GABAAB.e_pas = params['e_pas']
            GABAAB.e_GABAA = params['GABAA_e']
            GABAAB.tau_r_GABAA = params['tau_r_GABAA']
            GABAAB.tau_d_GABAA = params['tau_d_GABAA']
            GABAAB.GABAB_ratio = params['GABAB_ratio']
            GABAAB_ncon = h.NetCon(None, GABAAB)
            GABAAB_ncon.weight[0] = params['GABAA_default_conductance']

        inh_synapses.append(GABAAB)
        inh_netcons.append(GABAAB_ncon)
        
    syn_df = pd.DataFrame({'segments':all_segments,'seg_lens':seg_lens, 'exc_synapses':exc_synapses, 'exc_netcons':exc_netcons,
                        'inh_synapses':inh_synapses, 'inh_netcons':inh_netcons})

    logger.info(f"Setting temperature to be {params['celsius']} degree celsius")
    h.celsius = params['celsius']

    return syn_df