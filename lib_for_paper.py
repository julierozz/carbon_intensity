from pyDOE import *
from pandas import read_csv,DataFrame
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
from scipy.interpolate import interp1d

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
	
def create_elec_comit_table(elec_comit,subeleclife,myyears):
	elec_comit_table = DataFrame(columns = myyears, index=subeleclife)
	for eleclife in subeleclife:
		if eleclife in [60,50,40,30,20]:
			elec_comit_scenar = get_elec_comit(int(eleclife),myyears,elec_comit)
		else:
			elec_comit_scenar = np.zeros(len(myyears))
			for y in range(len(myyears)):
				elec_comit_scenar[y] = float(interp1d([np.floor(eleclife/10)*10,np.floor(eleclife/10)*10+10], [get_elec_comit(int(np.floor(eleclife/10)*10),myyears,elec_comit)[y],get_elec_comit(int(np.floor(eleclife/10)*10+10),myyears,elec_comit)[y]])(eleclife))
		elec_comit_table.loc[eleclife,:] = elec_comit_scenar
	return elec_comit_table
	
def calc_all_comit(IC_growth,GDPgrate,eleclife,myyears,elec_comit_table,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener,carb_int_ref,gdp_pc_ref,poprate,pop_ref):
	
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
	
	elec_comit_scenar = elec_comit_table.loc[eleclife,[str(y) for y in new_years]]

	co2_energy_not_power=co2_ener-elec_comit_scenar[0]
	enernotpower_com=calc_comit_e(eleclife,co2_energy_not_power,new_years)
	industry_com=calc_comit_e(induslife,co2_industry,new_years)
	tertiary_com=calc_comit_e(tertlife,co2_tertiary,new_years)
	transport_com=calc_comit_e(transplife,co2_transport,new_years)
	tot_comit=elec_comit_scenar+enernotpower_com+transport_com+tertiary_com+industry_com
	return tot_comit

def calc_new_intens(budget,GDPgrate,IC_growth,carb_int_ref,gdp_pc_ref,poprate,pop_ref,eleclife,myyears,elec_comit,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener,kintrate,kintref):
	if ini_year>2013:
		bau_years=[i for i in myyears if i<=ini_year]
		new_years=[i for i in myyears if i>=ini_year]
		co2_proj=calc_bau_emissions(GDPgrate,IC_growth,carb_int_ref,bau_years,gdp_pc_ref,poprate,pop_ref)
		already_emitted=10**(-6)*sum(co2_proj)
		pop_ref=pop_ref*(1+poprate)**(ini_year-2013)
		gdp_pc_ref=gdp_pc_ref*(1+GDPgrate)**(ini_year-2013)
		gdp_pc_ref=gdp_pc_ref*(1+GDPgrate)**(ini_year-2013)
		# kintref=kintref*(1+kintrate)**(ini_year-2013)
	else:
		new_years=myyears
		already_emitted=0
	new_budget=budget-10**(-9)*already_emitted
	
	total_comit=10**(-9)*calc_all_comit(IC_growth,GDPgrate,eleclife,myyears,elec_comit,co2_industry,co2_tertiary,co2_transport,induslife,tertlife,transplife,ini_year,co2_ener,carb_int_ref,gdp_pc_ref,poprate,pop_ref)
	
	remain_e=new_budget-sum(total_comit)
	gdp_tot=calc_proj(poprate,pop_ref,new_years)*calc_proj(GDPgrate,gdp_pc_ref,new_years)
	
	# k_over_g=calc_proj(kintrate,kintref,new_years)
	# ktot = gdp_tot*k_over_g
	
	gdp_comit=np.array([gdp_tot[0]]*len(gdp_tot))*(total_comit/np.array(len(total_comit)*[total_comit[0]]))
	gdp_new=gdp_tot-gdp_comit
	# k_new=gdp_new*k_over_g
	
	#in gco2/usd
	carb_intens_new_G=10**15*remain_e/sum(gdp_new)
	carb_intens=10**15*budget/sum(gdp_tot)
	# k_over_g_av=np.mean(k_over_g)
	return carb_intens_new_G,carb_intens,remain_e,total_comit
	
def create_scenarios(ranges,numCases):
	numUncertainties=len(ranges)
	lhsample= lhs(numUncertainties,numCases)
	scenarios=lhsample*np.diff(ranges[['min','max']].values).T+ranges['min'].values
	scenarios=DataFrame(scenarios,columns=ranges['variable'])
	return scenarios