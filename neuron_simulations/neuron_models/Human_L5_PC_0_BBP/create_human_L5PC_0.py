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
conduct_pas=2.950219236139216e-05
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
        seg.gNaTgbar_NaTg_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*6.711072225688051
    sec.insert("NaTg_persistent_somadend")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*0.002070682514148464
    sec.insert("NaTg_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*0.3956327902663066
    sec.insert("NaTg_persistent_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*0.004243258158998363
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=1.7334904939749267
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.00015585143031312106
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.005698047436291208
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.004247403344044463
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=266.0277644614975
    sec.gamma_CaDynamics_DC0=0.03908462918779915                
    sec.insert("KdShu2007")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gkbar_KdShu2007=((distance_from_soma-5)/46)*0.0022278199507432817
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
    sec.gNaTgbar_NaTg_somadend=2.004134390154651
    sec.insert("NaTg_persistent_somadend")
    sec.gNaTgbar_NaTg_persistent_somadend=0.0004700638368952195
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.1362882311790084
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.08037452740268843
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.18626157152349043
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.00048188676598420635
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.00034174955486781025
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.0620209193043439
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=269.18035566741895
    sec.gamma_CaDynamics_DC0=0.03553581136456214
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*8.948598901729844e-05)*0.00018932544670050694
    sec.ena=erev_na
    sec.ek=erev_k

for sec in apical:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=-0.22700435270779185
    sec.vshifth_NaTg=7.659204959572315
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.0723066835253172)*4.999999858590343e-10
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.011561869342976261
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.009492902912562037
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=4.9999997198124646e-11
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0077551826130229234
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.02803231273927677
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*8.948598901729844e-05)*0.00018932544670050694
    sec.ena=erev_na
    sec.ek=erev_k

for sec in basal:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=-0.22700435270779185
    sec.vshifth_NaTg=7.659204959572315
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.0723066835253172)*4.999999858590343e-10
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.011561869342976261
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.009492902912562037
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=4.9999997198124646e-11
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0077551826130229234
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.02803231273927677
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*8.948598901729844e-05)*0.00018932544670050694
    sec.ena=erev_na
    sec.ek=erev_k
    
    
    
print('model created!')