#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
**** NOTE ****
This python file is for generating the dataset for all four parameters of Chemical Reactor for Individual parameter
control chart and all parameter reactor stability index.
**** DataFrames ****
1. 'runtime_stability_index' DataFrame contains parameters values for Chemical Reactor Stability Index. Kindly refer to Cell No.25
2. 'df_control_chart_chemical_reactor' DataFrame contains values of all the parameters necessary for individual control charts
Kindly refer to Cell No.27
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv('chemical_reactor_stability_index_dataset.csv')
data



mass_temp_mean=data['Temperature_of_Mass'].mean()
mass_temp_std=data['Temperature_of_Mass'].std()
mass_temp_ucl=mass_temp_mean+(3*mass_temp_std)
mass_temp_lcl=mass_temp_mean-(3*mass_temp_std)
print('mass_temp_mean',mass_temp_mean,'mass_temp_std',mass_temp_std,'\nmass_temp_ucl',mass_temp_ucl,'mass_temp_lcl',mass_temp_lcl)
jacket_temp_mean=data['Jacket_Return_Temperature'].mean()
jacket_temp_std=data['Jacket_Return_Temperature'].std()
jacket_temp_ucl=jacket_temp_mean+(3*jacket_temp_std)
jacket_temp_lcl=jacket_temp_mean-(3*jacket_temp_std)
print('jacket_temp_mean',jacket_temp_mean,'jacket_temp_std',jacket_temp_std,'\njacket_temp_ucl',jacket_temp_ucl,'jacket_temp_lcl',jacket_temp_lcl)
pressure_mean=data['Chemical_Reactor_Pressure'].mean()
pressure_std=data['Chemical_Reactor_Pressure'].std()
pressure_ucl=pressure_mean+(3*pressure_std)
pressure_lcl=pressure_mean-(3*pressure_std)
print('pressure_mean',pressure_mean,'pressure_std',pressure_std,'\npressure_ucl',pressure_ucl,'pressure_lcl',pressure_lcl)
rpm_mean=data['Chemical_Reactor_RPM'].mean()
rpm_std=data['Chemical_Reactor_RPM'].std()
rpm_ucl=rpm_mean+(3*rpm_std)
rpm_lcl=rpm_mean-(3*rpm_std)
print('rpm_mean',rpm_mean,'rpm_std',rpm_std,'\nrpm_ucl',rpm_ucl,'rpm_lcl',rpm_lcl)



list_mean_mass_temp=[]
list_mean_jacket_temp=[]
list_mean_pressure=[]
list_mean_rpm=[]
list_ucl_mass_temp=[]
list_ucl_jacket_temp=[]
list_ucl_pressure=[]
list_ucl_rpm=[]
list_lcl_mass_temp=[]
list_lcl_jacket_temp=[]
list_lcl_pressure=[]
list_lcl_rpm=[]
for i in range(len(data)):
    list_mean_mass_temp.append(mass_temp_mean)
    list_mean_jacket_temp.append(jacket_temp_mean)
    list_mean_pressure.append(pressure_mean)
    list_mean_rpm.append(rpm_mean)
    list_ucl_mass_temp.append(mass_temp_ucl)
    list_ucl_jacket_temp.append(jacket_temp_ucl)
    list_ucl_pressure.append(pressure_ucl)
    list_ucl_rpm.append(rpm_ucl)
    list_lcl_mass_temp.append(mass_temp_lcl)
    list_lcl_jacket_temp.append(jacket_temp_lcl)
    list_lcl_pressure.append(pressure_lcl)
    list_lcl_rpm.append(rpm_lcl)



# Appling PCA for dimensionality reduction 
#Importing the PCA module
from sklearn.decomposition import PCA
pca = PCA()
#Doing the PCA on the whole data
pca.fit(data)
print('Components:')
print(pca.components_)
print('Explained Variance Ratio:')
print(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_)
plt.show()


# The PCA sugessests that from the PCA Matrix Produced the digonal elemensts 3.14789472e-04, 2.35054274e-02, 2.35012597e-02
# -3.60860636e-04. This suggests that the importance is in order Temperature_of_Mass, Jacket_Return_Temperature, 
# Chemical_Reactor_Pressure, Chemical_Reactor_RPM
# The stability index now can be measured by asscociating weights in the order of importance.
# chemical_stability_index=Temperature_of_Mass*1+Jacket_Return_Temperature*0.75+Chemical_Reactor_Pressure*0.5+Chemical_Reactor_RPM*0.0001
print('Means of Original Dataset-\n',data.mean())
# Ideal stability index value  estabilised form an mean of parameter for an ideal experiment
Temperature_of_Mass         =  24.050419
Jacket_Return_Temperature   =  64.454809
Chemical_Reactor_Pressure   = 150.100370
Chemical_Reactor_RPM        = 1054.811111
ideal_chemical_stability_index=(1-(Temperature_of_Mass*3.14789472e-04+Jacket_Return_Temperature*2.35054274e-02+Chemical_Reactor_Pressure*2.35012597e-02+Chemical_Reactor_RPM*-3.60860636e-04)/4)*100
print('ideal_chemical_stability_index - ',ideal_chemical_stability_index)


minute_5_all_run_time_index=[]
ideal_stability_index=[]
for j in range(int(180/5)): 
    means_list=[]
    # Mean of first 5 minute
    minute_5_mean_data=data.loc[:5*(j+1)]
    means_list=list(minute_5_mean_data.mean())
    print('Mean of first 5 minute -\n',means_list)
    # Calculating Stability Index for first 5 Minutes
    min_5_chemical_stability_index=(1-(means_list[0]*3.14789472e-04+means_list[1]*2.35054274e-02+means_list[2]*2.35012597e-02+means_list[3]*-3.60860636e-04)/4)*100
    print('The Stability Index - ',min_5_chemical_stability_index)
    minute_5_all_run_time_index.append(min_5_chemical_stability_index)
    ideal_stability_index.append(ideal_chemical_stability_index)



multi_dict={'ideal_stability_index':ideal_stability_index,'runtime_stability_index':minute_5_all_run_time_index}
runtime_stability_index=pd.DataFrame(multi_dict)
runtime_stability_index.to_excel('reactor_stability_index_dataset.xlsx')



runtime_stability_index.plot.line()



data_dict={'Temperature_of_Mass':data['Temperature_of_Mass'],'mean_mass_temp':list_mean_mass_temp,'ucl_mass_temp':list_ucl_mass_temp,'lcl_mass_temp':list_lcl_mass_temp,
           'Jacket_Return_Temperature':data['Jacket_Return_Temperature'],'mean_jacket_temp':list_mean_jacket_temp,'ucl_jacket_temp':list_ucl_jacket_temp,'lcl_jacket_temp':list_lcl_jacket_temp,
           'Chemical_Reactor_Pressure':data['Chemical_Reactor_Pressure'],'mean_pressure':list_mean_pressure,'ucl_pressure':list_ucl_pressure,'lcl_pressure':list_lcl_pressure,
           'Chemical_Reactor_RPM':data['Chemical_Reactor_RPM'],'mean_rpm':list_mean_rpm,'ucl_rpm':list_ucl_rpm,'lcl_rpm':list_lcl_rpm}
df_control_chart_chemical_reactor=pd.DataFrame(data_dict)
df_control_chart_chemical_reactor.to_excel('control_chart_dataset.xlsx')



df_mass_temp=df_control_chart_chemical_reactor.iloc[:,0:4]
df_jacket_temp=df_control_chart_chemical_reactor.iloc[:,4:8]
df_pressure=df_control_chart_chemical_reactor.iloc[:,8:12]
df_rpm=df_control_chart_chemical_reactor.iloc[:,12:16]



df_mass_temp.plot.line()



df_jacket_temp.plot.line()



df_pressure.plot.line()



df_rpm.plot.line()
