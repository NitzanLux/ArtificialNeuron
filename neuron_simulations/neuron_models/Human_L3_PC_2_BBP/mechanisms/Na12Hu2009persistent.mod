:Reference :Hu et al., Nature Neuroscience, 2009

: Adapted by Mickael Zbili @ BBP, 2020:
: LJP: corrected in the paper!

NEURON	{
	SUFFIX Na12Hu2009persistent
	USEION na READ ena WRITE ina
	RANGE gNa12bar, gNa12, ina, vshiftm, slopem
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNa12bar = 0.00001 (S/cm2)
	vshiftm = 6 (mV)
	slopem = 7
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNa12	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNa12 = gNa12bar*m*m*m
	ina = gNa12*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-23)/10)

  UNITSOFF
    if(v == (-43+vshiftm)){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- (-43+vshiftm)))/(1-(exp(-(v- (-43+vshiftm))/slopem)))
		mBeta  = (0.124 * (-v + (-43+vshiftm)))/(1-(exp(-(-v + (-43+vshiftm))/slopem)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = (mAlpha/(mAlpha + mBeta))
	UNITSON
}
