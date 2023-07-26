TITLE HH KA anomalous rectifier channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
: Adapted from /usr/local/neuron/demo/release/khhchan.mod - squid 
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

NEURON {
	SUFFIX kANicoKmodified
	USEION k READ ek WRITE ik
	RANGE gkbar, ik, ek
	GLOBAL pinf, qinf, ptau, qtau,Aptau, Aqtau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
PARAMETER {
	v (mV)
	dt (ms)
	gkbar=.036 (mho/cm2) <0,1e9>
	Aptau = 1.38 (ms)
	Aqtau = 50 (ms)
	vhalf_p=-42
	vhalf_q=-90
	slope_p=13
	slope_q=10
}

ASSIGNED {
	ik (mA/cm2)
    ek (mV)
	pinf
	qinf
	ptau (ms)
	qtau (ms)
}

STATE {
	p q
}

INITIAL {
	rates(v)
	p = pinf
	q = qinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ik = gkbar*p*q*(v - ek)
}

DERIVATIVE states {
	rates(v)
	p' = (pinf - p)/ptau
	q' = (qinf - q)/qtau
}

PROCEDURE rates(v(mV)) {
	TABLE pinf, qinf, ptau, qtau FROM -100 TO 100 WITH 200
	ptau = Aptau
	qtau = Aqtau
	pinf = 1/(1 + exp(-(v*1(/mV) - vhalf_p)/slope_p))
	qinf = 1/(1 + exp((v*1(/mV) - vhalf_q)/slope_q))
}

