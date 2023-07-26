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
erev_pas=-94.9563201439754
conduct_pas=2.8369790393442582e-05
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
        seg.gNaTgbar_NaTg_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*7.2397184772775764
    sec.insert("NaTg_persistent_somadend")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_somadend=(1/(1+np.exp((distance_from_soma-25)/3)))*0.0775070498789227
    sec.insert("NaTg_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*3.7976105366880093
    sec.insert("NaTg_persistent_axon")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](1), seg)
        seg.gNaTgbar_NaTg_persistent_axon=(1-(1/(1+np.exp((distance_from_soma-20)/2.5))))*0.00022176193097255686
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.8322622404628857
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.000605297638472977
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=1.7607979459766977e-05
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.014257001766941627
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=298.25043682253784
    sec.gamma_CaDynamics_DC0=0.005769064857487776                
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
    sec.gNaTgbar_NaTg_somadend=0.38666443014445173
    sec.insert("NaTg_persistent_somadend")
    sec.gNaTgbar_NaTg_persistent_somadend=0.001056491726707437
    sec.insert("K_Pst")
    sec.gK_Pstbar_K_Pst=0.14717382471618617
    sec.insert("K_Tst")
    sec.gK_Tstbar_K_Tst=0.0015275186629560913
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.008333532049351056
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.00025986566478844355
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.009103803997757726
    sec.insert("SK_E2")
    sec.gSK_E2bar_SK_E2=0.0002331056847965987
    sec.insert("CaDynamics_DC0")
    sec.decay_CaDynamics_DC0=30.55156655783756
    sec.gamma_CaDynamics_DC0=0.024185231677343908
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002311616633417006)*0.00016645133005209944
    sec.ena=erev_na
    sec.ek=erev_k

for sec in apical:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=8.940289942422444
    sec.vshifth_NaTg=-0.08874076326942593
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.08088329206599906)*0.008104723852621898
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.0016774062390637245
    #sec.insert("K_Pst")
    #sec.gK_Pstbar_K_Pst=0.009492902912562037
    #sec.insert("K_Tst")
    #sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.00034520624052486844
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0003200031729789866
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.07412668470559458
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002311616633417006)*0.00016645133005209944
    sec.ena=erev_na
    sec.ek=erev_k

for sec in basal:
    sec.Ra=ax_res
    sec.cm=capa_mb*2
    sec.insert('pas')
    sec.g_pas=conduct_pas
    sec.e_pas=erev_pas
    sec.insert("NaTg")
    sec.vshiftm_NaTg=8.940289942422444
    sec.vshifth_NaTg=-0.08874076326942593
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gNaTgbar_NaTg=np.exp(distance_from_soma*-0.08088329206599906)*0.008104723852621898
    sec.insert("SKv3_1")
    sec.gSKv3_1bar_SKv3_1=0.0016774062390637245
    #sec.insert("K_Pst")
    #sec.gK_Pstbar_K_Pst=0.009492902912562037
    #sec.insert("K_Tst")
    #sec.gK_Tstbar_K_Tst=0.06286967833045491
    sec.insert("Ca_HVA2")
    sec.gCa_HVAbar_Ca_HVA2=0.00034520624052486844
    sec.insert("Ca_LVAst")
    sec.gCa_LVAstbar_Ca_LVAst=0.0003200031729789866
    sec.insert("CaDynamics_DC0")
    sec.gamma_CaDynamics_DC0=0.07412668470559458
    sec.insert("Ih")
    for seg in sec:
        distance_from_soma = h.distance(cell.soma[0](0), seg)
        seg.gIhbar_Ih=np.exp(distance_from_soma*0.002311616633417006)*0.00016645133005209944
    sec.ena=erev_na
    sec.ek=erev_k
    
    
    
print('model created!')