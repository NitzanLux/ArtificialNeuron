import neuron
from neuron import h, gui
from neuron.units import ms, mV
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# os.system("rm -rf x86_64")
# os.system("nrnivmodl mechanisms")

h.load_file('stdrun.hoc')
neuron.h.stdinit();
h.nrn_load_dll('x86_64/nrnmech.dll')




"""
Import morphology and create cell
"""
h.load_file('import3d.hoc')

from pathlib import Path
script_dir = Path( __file__ ).parent.absolute()

class MyCell:
    def __init__(self):
        morph_reader = h.Import3d_Neurolucida3()
        morph_reader.input(f'{script_dir}/morphologies/2057_H21_29_197_11_01_03_metcontour.asc')
        i3d = h.Import3d_GUI(morph_reader, 0)
        i3d.instantiate(self)

cell = MyCell()


"""
create sectionlist and put the sections into it
"""
apical = h.SectionList()
for sec in cell.apic:
    apical.append(sec)
somatic = h.SectionList()
for sec in cell.soma:
    somatic.append(sec)
basal = h.SectionList()
for sec in cell.dend:
    basal.append(sec)
axonal = h.SectionList()
for sec in cell.axon:
    axonal.append(sec)
axon_initial_segment = h.SectionList()
myelinated = h.SectionList()




"""
Replace axon with a to 45 micrometers axon initial segment followed by 1000 micrometers myelinated section
"""
ZERO = 1e-6
length_ais=45
delta=0.1
taper_scale=1
taper_strength=0
myelin_diameter=1
nseg_frequency=5

def taper_function(distance, strength, taper_scale, terminal_diameter, scale=1.0):
    """Function to model tappered AIS."""
    return strength * np.exp(-distance / taper_scale) + terminal_diameter * scale
    
# connect basal and apical dendrites to soma at loc=0
for section in basal:
    sref = neuron.h.SectionRef(sec=section)
    if sref.parent==cell.soma[0]:
        seg_connection=neuron.h.section_orientation(sec=section)
        neuron.h.disconnect(sec=section)
        section.connect(cell.soma[0], 0.0, seg_connection)
    section.nseg = int(section.L / 27.88) + 1
for section in apical:
    sref = neuron.h.SectionRef(sec=section)
    if sref.parent==cell.soma[0]:
        seg_connection=neuron.h.section_orientation(sec=section)
        neuron.h.disconnect(sec=section)
        section.connect(cell.soma[0], 0.0, seg_connection)
    section.nseg = int(section.L / 27.88) + 1

# delete all axonal sections
for section in axonal:
    neuron.h.delete_section(sec=section)

# set hillock section
hillock=neuron.h.Section(name='hillock')
nseg_hillock = 1 + 2 * int(delta / nseg_frequency)
diameters_hillock = taper_function(
    np.linspace(0, delta, nseg_hillock), taper_strength, taper_scale, myelin_diameter
)
count = 0
section = hillock
section.nseg = nseg_hillock
section.L = delta
for seg in section:
    seg.diam = diameters_hillock[count]
    count += 1
somatic.append(sec=section)

# set ais section
ais=neuron.h.Section(name='ais')
nseg_ais = 1 + 2 * int(length_ais / nseg_frequency)
diameters_ais = taper_function(
    np.linspace(delta, length_ais+delta, nseg_ais), taper_strength, taper_scale, myelin_diameter
)
count = 0
section = ais
section.nseg = nseg_ais
section.L = length_ais
for seg in section:
    seg.diam = diameters_ais[count]
    count += 1
axon_initial_segment.append(sec=section)

# set myelinated axon
myelin=neuron.h.Section(name='myelin')
section = myelin
section.nseg = 5
section.L = 1000
section.diam = myelin_diameter
myelinated.append(sec=section)

# connect soma/hillock/ais/myelin
hillock.connect(cell.soma[0], 1.0, 0.0)
ais.connect(hillock, 1.0, 0.0)
myelin.connect(ais, 1.0, 0.0)


        
"""
Insert channels and define parameters of the different sections
"""
ax_res=100
capa_mb=1
erev_pas=-94.9999999825
conduct_pas=3.455523415875226e-05
erev_na=50
erev_k=-90

for sec in myelinated:
    sec.Ra=ax_res
    sec.cm=capa_mb/11
    sec.insert('pas')
    sec.g_pas=conduct_pas/11
    sec.e_pas=erev_pas

for sec in axon_initial_segment:
    sec.Ra=ax_res
    sec.cm=capa_mb
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas 
    sec.insert("NaTg_somadend")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*4.02797589111084
    sec.insert("NaTg_persistent_somadend")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*0.0376795330262421
    sec.insert("NaTg_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*3.999999886872274e-09
    sec.insert("NaTg_persistent_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*0.0056282103783044934
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=1.505113273938676
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.0005234242278554381
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0005630874254744754
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.03751091160941056
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=207.77665278796587
    sec.gamma_CaDynamics_DC0=0.02189177222608897                
    sec.insert("KdShu2007")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gkbar_KdShu2007=((distance_from_soma-5)/46)*1.4999999020659516e-09
    sec.insert("Kv7")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gKv7bar_Kv7=(1.1*np.exp(distance_from_soma*0.086))*9.999999717180685e-10
    sec.ena=erev_na
    sec.ek=erev_k

for sec in somatic:
    sec.Ra=ax_res
    sec.cm=capa_mb
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg_somadend")
    sec.gNaTgbar_NaTg_somadend=2.04229952893085
    sec.insert("NaTg_persistent_somadend")
    sec.gNaTgbar_NaTg_persistent_somadend=0.003741741263298924
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.15519803624257755
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.058550775998882774
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.6460660318137572
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.0009999999995
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=9.665139311318145e-05
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.06655907465840114
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=160.1899308529783
    sec.gamma_CaDynamics_DC0=0.017154978738641316
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*2.4999997731700585e-12)*0.0001999999999
    sec.ena=erev_na
    sec.ek=erev_k

for sec in apical:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=2.8011897621983746
    sec.vshifth_NaTg=-0.7282993881671578
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.05026276278833691)*0.592409094898815
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.02159220873836165
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.02066569712876551
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.09881194878438417
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.002470326748577306
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.006322790762659361
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.04010668923288436
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*2.4999997731700585e-12)*0.0001999999999
    sec.ena=erev_na
    sec.ek=erev_k

for sec in basal:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=2.8011897621983746
    sec.vshifth_NaTg=-0.7282993881671578
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.05026276278833691)*0.592409094898815
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.02159220873836165
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.02066569712876551
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.09881194878438417
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.002470326748577306
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.006322790762659361
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.04010668923288436
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*2.4999997731700585e-12)*0.0001999999999
    sec.ena=erev_na
    sec.ek=erev_k
    
    
    
print('model created!')