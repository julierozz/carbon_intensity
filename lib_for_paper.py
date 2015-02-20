from pyDOE import *
from pandas import read_csv,DataFrame
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def calc_proj(grate,ref,myyears):
	out=np.array([ref*(1+grate)**(i-myyears[0]) for i in myyears])
	return out
	
def calc_bau_emissions(GDPgrate,IC_growth,carb_int_ref,myyears,gdp_pc_ref,poprate,pop_ref):
	intensC_bau=calc_proj(IC_growth,carb_int_ref,myyears)
	gdp_pc_bau=calc_proj(GDPgrate,gdp_pc_ref,myyears)
	pop=calc_proj(poprate,pop_ref,myyears)
	co2_proj=intensC_bau*pop*gdp_pc_bau
	return co2_proj
	
def calc_comit_e(lifetime,ini_co2,myyears):
	dep_rate=1/lifetime
	com_e=np.array([(1-dep_rate*(i-myyears[0]))*ini_co2 for i in myyears])
	com_e[com_e<0]=0
	return com_e
	
def get_elec_comit(lifetime,myyears,elec_comit):
	thestr="life{}years".format(lifetime)
	yy=list(set(myyears).intersection(elec_comit['year']))
	yy=np.sort(yy)
	elec_comit_scenar=np.array([elec_comit.ix[elec_comit['year']==i,thestr].values[0] for i in yy]+[0 for i in list(set(myyears)-set(elec_comit['year']))])
	return elec_comit_scenar
	
def calc_all_comit(IC_growth,GDPgrate,eleclife,myyears,elec_comit,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener,carb_int_ref,gdp_pc_ref,poprate,pop_ref):
	
	if ini_year>2013:
		bau_years=[i for i in myyears if i<=ini_year]
		new_years=[i for i in myyears if i>=ini_year]
		co2_proj=calc_bau_emissions(GDPgrate,IC_growth,carb_int_ref,bau_years,gdp_pc_ref,poprate,pop_ref)
		co2_industry=co2_industry*co2_proj[-1]/co2_proj[0]
		co2_tertiary=co2_tertiary*co2_proj[-1]/co2_proj[0]
		co2_transport=co2_transport*co2_proj[-1]/co2_proj[0]
		co2_ener=co2_ener*co2_proj[-1]/co2_proj[0]
	else:
		new_years=myyears
		
	elec_comit_scenar=get_elec_comit(eleclife,new_years,elec_comit)
	co2_energy_not_power=co2_ener-elec_comit_scenar[0]
	enernotpower_com=calc_comit_e(eleclife,co2_energy_not_power,new_years)
	industry_com=calc_comit_e(induslife,co2_industry,new_years)
	tertiary_com=calc_comit_e(tertlife,co2_tertiary,new_years)
	transport_com=calc_comit_e(transplife,co2_transport,new_years)
	tot_comit=elec_comit_scenar+enernotpower_com+transport_com+tertiary_com+industry_com
	return tot_comit

def calc_new_intens(budget,GDPgrate,IC_growth,carb_int_ref,gdp_pc_ref,poprate,pop_ref,eleclife,myyears,elec_comit,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener):
	if ini_year>2013:
		bau_years=[i for i in myyears if i<=ini_year]
		new_years=[i for i in myyears if i>=ini_year]
		co2_proj=calc_bau_emissions(GDPgrate,IC_growth,carb_int_ref,bau_years,gdp_pc_ref,poprate,pop_ref)
		already_emitted=sum(co2_proj)
	else:
		new_years=myyears
		already_emitted=0
	new_budget=10**9*budget-already_emitted
	
	total_comit=calc_all_comit(IC_growth,GDPgrate,eleclife,myyears,elec_comit,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener,carb_int_ref,gdp_pc_ref,poprate,pop_ref)
	
	remain_e=new_budget-sum(total_comit)
	gdp_tot=calc_proj(poprate,pop_ref,new_years)*calc_proj(GDPgrate,gdp_pc_ref,new_years)
	
	gdp_comit=np.array([gdp_tot[0]]*len(gdp_tot))*(total_comit/np.array(len(total_comit)*[total_comit[0]]))
	gdp_new=sum(gdp_tot-gdp_comit)
	#in kgco2/usd
	carb_intens_new_GDP=10**3*remain_e/gdp_new
	return carb_intens_new_GDP,remain_e,total_comit
	
def create_scenarios(ranges,numCases):
	numUncertainties=len(ranges)
	lhsample= lhs(numUncertainties,numCases)
	scenarios=lhsample*np.diff(ranges[['min','max']].values).T+ranges['min'].values
	scenarios=DataFrame(scenarios,columns=ranges['variable'])
	return scenarios