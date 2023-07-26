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
        morph_reader.input(f'{script_dir}/morphologies/1496_Thijs_22juni_slice1_cell3.asc')
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
    section.nseg = int(section.L / 13.11) + 1
for section in apical:
    sref = neuron.h.SectionRef(sec=section)
    if sref.parent==cell.soma[0]:
        seg_connection=neuron.h.section_orientation(sec=section)
        neuron.h.disconnect(sec=section)
        section.connect(cell.soma[0], 0.0, seg_connection)
    section.nseg = int(section.L / 13.11) + 1

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
erev_pas=-92.69791214875136
conduct_pas=2.8972423680476844e-05
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
        seg.gNaTgbar_NaTg_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*1.6024531675732185
    sec.insert("NaTg_persistent_somadend")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*3.9999996370720936e-11
    sec.insert("NaTg_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*7.567679132027485
    sec.insert("NaTg_persistent_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*0.016946756802529436
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=1.9588031755058182
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.0009404778391950109
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.009806414366917844
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.0049598569596299505
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=289.1724274267791
    sec.gamma_CaDynamics_DC0=0.006251148826260721                
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
    sec.gNaTgbar_NaTg_somadend=0.4072312829274436
    sec.insert("NaTg_persistent_somadend")
    sec.gNaTgbar_NaTg_persistent_somadend=0.00015520740583265663
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.15868796333637913
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.002620749262756142
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.14242965531801666
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=2.0541216009091208e-05
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0010645001542318351
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.026058090153186355
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=246.70262080594662
    sec.gamma_CaDynamics_DC0=0.006676098398609727
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002924721558313136)*0.00015816925841374092
    sec.ena=erev_na
    sec.ek=erev_k

for sec in apical:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=-0.28998938460648116
    sec.vshifth_NaTg=2.1543949436553733
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.060745622244231516)*0.19024987997268838
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.0007109203703121814
    #sec.insert("K_Pst")
    #sec.gK_Pstbar_K_Pst=0.009492902912562037
    #sec.insert("K_Tst")
    #sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.0007643699561447853
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.000445904524037379
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.017144408807856754
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002924721558313136)*0.00015816925841374092
    sec.ena=erev_na
    sec.ek=erev_k

for sec in basal:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=-0.28998938460648116
    sec.vshifth_NaTg=2.1543949436553733
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.060745622244231516)*0.19024987997268838
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.0007109203703121814
    #sec.insert("K_Pst")
    #sec.gK_Pstbar_K_Pst=0.009492902912562037
    #sec.insert("K_Tst")
    #sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.0007643699561447853
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.000445904524037379
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.017144408807856754
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002924721558313136)*0.00015816925841374092
    sec.ena=erev_na
    sec.ek=erev_k
    
    
    
print('model created!')