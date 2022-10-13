#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: lukefanguna
"""
#Import necessary packages
import os
import numpy as np
from numpy import newaxis
from scipy import signal
from functions import data2index,data2index_larger,calc_plotting_position
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
""" 
note : matlab needs to clear variables
but python does need to?? check again

#reset variable func from https://gist.github.com/stsievert/8655158355b7155b2dd8#file-ssreset-py-L1
#from IPython import get_ipython
#def __reset__(): get_ipython().magic('reset -sf',24)

#reset variables
#__reset__()
"""
# training period is 2003-2013 ##
#this part of the code reads input data and creates smoothed climatology of wildfire burned data

""" SHOULD WE MAKE THE FILES UPLOADABLE SO IT IS EASIER TO ACCESS """
FDEO_DIR = os.path.dirname(os.path.dirname(__file__))

#importing the land cover file (lc1.csv)
lc1_path = os.path.join(FDEO_DIR, 'data', "lc1.csv")#file path
lc1 = np.loadtxt(lc1_path, delimiter=",")#separates txt
#lc1 = lc1.reshape((112,244,132));
"""
% ID for each Land cover type
%1: Lake
%2: Developed/Urban
%3: Barren Land  
%4: Deciduous
%5: Evergreen
%6: Mixed Forest
%7: Shrubland
%8: Herbaceous
%9: Planted/Cultivated
%10: Wetland
%15: ocean
"""
## where data came from
#soil moisture (sm) data from 2003-2013
sm_path = os.path.join(FDEO_DIR, 'data', "sm_20032013.csv");
sm_20032013 = np.loadtxt(sm_path, delimiter=",");
sm_20032013 = sm_20032013.reshape((112,244,132));
#vapor pressure deficit (vpd) data from 2003-2013
vpd_path = os.path.join(FDEO_DIR, 'data', "vpd_20032013.csv");
vpd_20032013 = np.loadtxt(vpd_path, delimiter=",");
vpd_20032013 = vpd_20032013.reshape((112,244,132));
#enhanced vegetation index (EVI) data from 2003-2013
EVI_path = os.path.join(FDEO_DIR, 'data', "EVI_20032013.csv");
EVI_20032013 = np.loadtxt(EVI_path, delimiter=",");
EVI_20032013 = EVI_20032013.reshape((112,244,132));
""" what is fire data """
#Fire data from 2003-2013
fire_path = os.path.join(FDEO_DIR, 'data', "firemon_tot_size.csv");
firemon_tot_size = np.loadtxt(fire_path, delimiter=",");
firemon_tot_size = firemon_tot_size.reshape((112,244,132));
#size = firemon_tot_size.size()
#calculate the fire climatology for each month##

#split the dim sizes for ease
mtrxshape = np.shape(firemon_tot_size);
firemon_tot_sizeX = mtrxshape[0];
firemon_tot_sizeY = mtrxshape[1];
firemon_tot_sizeZ = mtrxshape[2];

firemon_tot_size_climatology = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,12));
for i in range(firemon_tot_sizeX):#x
    for j in range(firemon_tot_sizeY):#y
        for k in range(12):
            
            firemon_tot_size_climatology[i][j][k]= np.mean(firemon_tot_size[i][j][k:firemon_tot_sizeZ:12]);

#spatially smooth fire data using a 3 by 3 filter
smoothfilter = np.ones((3,3));

firemon_tot_size_climatology_smoothed_3 = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,12));
for h in range(len(firemon_tot_size_climatology)):
    rotatingARR = firemon_tot_size_climatology[:][:][h];
    rotator = np.rot90(rotatingARR);
    firemon_tot_size_climatology_smoothed_3[:][:][h] = signal.convolve2d(firemon_tot_size_climatology[:][:][h], rotator, mode='same');
    
firemon_tot_size_climatology_smoothed_3=firemon_tot_size_climatology_smoothed_3/(len(smoothfilter[0])*len(smoothfilter[1]));

#This part of the code creates a regression model for each LC type based on
#the "best" drought indicator (DI) and then creates a historical record of
#probabilistic and categorical wildfire prediction and observation data


"""

define the best DI variable to burned area for each LC type
function data2index is used to derive DI of SM and EVI
function data2index_larger is used to derive DI of VPD

"""

#Deciduous DI
sc_drought = 1; #month range
sm_new_drought=data2index(sm_20032013,sc_drought);
Deciduous_best_BA = sm_new_drought;

#Shrubland DI
sc_drought = 1;
EVI_drought=data2index(EVI_20032013,sc_drought);
Shrubland_best_BA = EVI_drought;

#Evergreen DI
sc_drought = 3;
vpd_new_drought = data2index_larger(vpd_20032013,sc_drought);
Evergreen_best_BA = vpd_new_drought;

#Herbaceous DI
Herbaceous_best_BA = vpd_new_drought;

#Wetland DI
sc_drought = 3;
sm_new_drought = data2index(sm_20032013,sc_drought);
Wetland_best_BA = sm_new_drought;



#vector of land-cover IDs according to below 
lctypemat = [4, 5, 7, 8, 10];

#1: Lake
#2: Developed/Urban
#3: Barren Land  
#4: Deciduous       []
#5: Evergreen       []
#6: Mixed Forest
#7: Shrubland       []
#8: Herbaceous
#9: Planted/Cultivated
#10: Wetland
#15: ocean

#set initial dimentions of prediction probabilistic matrix
fire_pred_ini = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,firemon_tot_sizeZ));

#Build the prediction model
#One model for each land cover type
# TODO: EV: This will loop from 0 to 4, and at the end of the loop will leave
#  var_for_forecast set to Herbaceous_best_BA
for lc_co in range(5):
    if  (lc_co == 1):
        var_for_forecast=Deciduous_best_BA;
    elif (lc_co == 2):
        var_for_forecast=Evergreen_best_BA;
    elif (lc_co==3):
        var_for_forecast=Shrubland_best_BA;
    elif (lc_co==4):
        var_for_forecast=Herbaceous_best_BA;
    elif (lc_co==5):
        var_for_forecast=Wetland_best_BA;


#first build the regression model for the LC Type
#initial parameters

# TODO: EV: Again, all of this should be inside the loop defining a LC type

# TODO: EV: 242353 far less than the size of firemon_tot. Will run into a index error with 'm'.
#  How was this number come up with?
mat = np.empty((2,242353),dtype=float);#initial array
m = 0;#Burned area for each LC. 1-5 is each diff one
lead = 1; #lead time
for i in range(firemon_tot_sizeX):#x
    for j in range(firemon_tot_sizeY):#y

        # TODO: EV: Again lc_co will always be 4 in this loop
        if (lc1[i][j]==lctypemat[lc_co]):
            print(True)
            for k in range(firemon_tot_sizeZ):
                #leave the first 3-month empty to account for 2-month lead model development
                if (k - lead < 1):
                    mat[0][m] = np.nan;
                    m=m+1;
                else:
                    #drought index    
                    mat[0][m]=var_for_forecast[i][j][k-lead];
                    #fire burned area
                    mat[1][m]=firemon_tot_size[i][j][k];
                    m=m+1;

# TODO: EV: Again, all of this should be inside the loop defining a LC type?

DImat = mat[0];
BAmat = mat[1];
np.nan_to_num(DImat,False,999);
np.nan_to_num(BAmat,False,999);
idx_nan_1 = np.where(mat[0] == 999);
idx_nan_2 = np.where(mat[1] == 999);
idx_nan_3 = np.concatenate((idx_nan_1, idx_nan_2), axis = None);#combines NaN values/stacks them
idx_nan_4 = np.unique(idx_nan_3);
mat = np.column_stack((DImat,BAmat));
mat = np.delete(mat, idx_nan_4, axis=0);


# bar plots to derive the regression model
# define number of bins
xbin = 10;

# derive min and max of DI (drought indicator)
min_1 = np.min(mat[0]);
max_1 = np.max(mat[0]);

# derive vector of bins based on DI data
varbin = np.arange(-1.6414,1.9697,.3283);#-1.6414->1.6414 in inc of .3283

# find observations in each bin
k = 0;
sample_size = np.empty((11,1));
fire_freq_ave = np.empty((11,1));
prob = np.empty((11,1));

for i in range(len(varbin)):
    
    # find observations in each bin
    idx3 = np.less_equal(np.logical_and(mat[0] >= varbin[i], mat[0]),(varbin[i] + ((np.max(mat[0])-np.min(mat[0]))/2)));
    # find DI in each bin
    var_range = mat[0][idx3];
    # get corresponding burned area in each bin
    fire_freq_range = mat[1][idx3];
    # calculate number of observations in each bin
    sample_size[k] = (len(idx3));
    # calculate sum burned area in each bin
    fire_freq_ave[k] = sum(fire_freq_range);# adds all
    # calculate probability of fire at each bin
    prob[k] = (fire_freq_ave[k]/sample_size[k]);
    k += 1;
    


# develop linear regression model
X = varbin;
Y = prob;
#idx_nan = np.where(np.logical_not(np.isnan(Y)));
idx_nan = np.isnan(Y);

## take out nan values in it
X = X[~idx_nan.flatten()];
Y = Y[~idx_nan.flatten()];

# fit the regression model
gofmat1 = np.polyfit(X,Y,2);##

# TODO: EV: Accessing lc_co, which at this point is stuck at 4. This and the above code
#   should all be indented? Once this is done, lc_co will equal 0 for one loop iteration,
#   and model_res and obs_res will be un-indexable.
#   And why index lc_co - 1?
# calculate model and observation values at each bin. Each row represents one LC Type
model_res = np.empty((lc_co,11));
obs_res = np.empty((lc_co,11));
for i in range(len(varbin)):
    # model at each bin
    model_res[lc_co-1][i] = gofmat1[1]*(varbin[i]**2)+gofmat1[1]*varbin[i]+gofmat1[2];
    # observation at each bin
    obs_res[lc_co-1][i] = prob[i];

del X
del Y

# now build a historical forecast matrix based on the developed regression model for each LC Type

#lc1 = np.empty((firemon_tot_sizeX, firemon_tot_sizeY));
for k in range(4,firemon_tot_sizeZ):
    
    for i in range(firemon_tot_sizeX):
        
        for j in range(firemon_tot_sizeY):

            # TODO: EV: Same thing here, lc_co is stuck at 4 outside of the loop
            if lc1[i][j] == lctypemat[lc_co]:
                
                fire_pred_ini[i][j][k] = gofmat1[0] * (var_for_forecast[i][j][k-lead] ** 2) + gofmat1[1] * var_for_forecast[i][j][k-lead] + gofmat1[2];


# TODO: EV: Now we exit the lc_co loop?
# build a correlation matrix and R2 matrix of goodness of fit for all
# models. Each row represents one LC Type
forrange = np.shape(model_res);
for i in range(forrange[0]):
    
    pmvec = model_res[i]#creates separate matrix
    povec = obs_res[i];#creates separate matrix
    # remove nan from observation
    idx_nan_4 = ~(obs_res[i][:] == 999);#finds NaN values
    pmvec = pmvec[idx_nan_4];#removes nan values
    povec = povec[idx_nan_4];#removes nan values
    # calculate correlation
    corr_mat = np.corrcoef(pmvec, povec);
    corr_mat = np.nan_to_num(corr_mat,False,999);
    # correlation vector of observation and model
    corr_vector = np.shape(model_res);
    corr_vector = np.empty(corr_vector[0]);
    corr_vector[i] = corr_mat[0][1];
    # R2 vector of observation and model
    r2_vector = np.shape(model_res);
    r2_vector = np.arange(r2_vector[0]);
    r2_vector[i] = corr_vector[i] ** 2;
    
# we now calculate anomalies for observation and predictions

# observation data
fire_obs_ini = firemon_tot_size;

## subtract prediction and observation from climatology to derive anomalies
fire_pred_ini_cate = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,firemon_tot_sizeZ));
fire_obs_ini_cate = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,firemon_tot_sizeZ));

obs_ini_split = np.dsplit(fire_obs_ini,11);
pred_ini_split = np.dsplit(fire_pred_ini,11);
for i in range(11):
    
    np.append(fire_obs_ini_cate,obs_ini_split[i]-firemon_tot_size_climatology_smoothed_3,axis = 2);
    np.append(fire_pred_ini_cate,pred_ini_split[i]-firemon_tot_size_climatology_smoothed_3,axis = 2);
    #fire_pred_ini_cate[:][:][i:i+12]=pred_ini_split[l]-firemon_tot_size_climatology_smoothed_3;



# Derive bias adjusted observation and prediction probabilities and categorical forecast for the entire time series
# distributin of prediction and observation come from Gringorten empirical disriburion function (empdis function) 


# vector of Land cover ID
lctypemat = [4, 5, 7, 8, 10];

# This section derives CDF of observation and prediction anomalies
# derive CDF for each land cover type and for each month. 
# Build probabilisitc prediction and observation matrices
val_new_obs_tot_1 = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,firemon_tot_sizeZ));
val_new_pred_tot_1 = np.empty((firemon_tot_sizeX,firemon_tot_sizeY,firemon_tot_sizeZ));
obs_split = np.dsplit(fire_obs_ini_cate,132);
pred_split = np.dsplit(fire_pred_ini_cate,132);
dimensions = np.shape(fire_pred_ini_cate)
# count = 0; 
for k in range(dimensions[0]):
    
    # matrix of observation and prediction anomalies for each month
    val_new_obs = obs_split[k]#fire_obs_ini_cate[:][:][k];
    val_new_pred = pred_split[k]#fire_pred_ini_cate[:][:][k];
    
    # derive CDF for each LC type
    for lc_co in range(len(lctypemat)):
        
        #derive observation and prediction anomalies for each LC Type
        idx_lc = np.equal(lc1,lctypemat[lc_co]); #creates a 1d array that fulfills cond
        
        #val_new_obs = val_new_obs; #makes the 2d into a 1d array
        #val_new_pred = val_new_pred.flatten(); #makes the 2d into a 1d array
        
        mat=val_new_obs[idx_lc]; # picks values from val_new_obs that fulfills cond
        mat1=val_new_pred[idx_lc]; # picks values from val_new_pred that fulfills cond
        
        # observation CDF
        y = calc_plotting_position(mat);

        # prediction CDF
        y1 = calc_plotting_position(mat1);
        # count +=1;
        # print(count)
        val_new_obs[idx_lc] = y;#.flatten();
        val_new_pred[idx_lc] = y;#1.flatten();

# build matrix of CDFs (probabilisitc prediction and observation matrices)
    
    val_new_obs_tot_1[k][:][:]=fire_obs_ini_cate[k][:][:];
    val_new_pred_tot_1[k][:][:]=fire_pred_ini_cate[k][:][:];

# TODO: EV: Should this loop be nested in the above loop which defines val_new_obs?
# build a loop for each LC Type
for lc_co in range(len(lctypemat)):
    
    #derive observation and prediction anomalies for each LC Type
    idx_lc = np.equal(lc1,lctypemat[lc_co]); #creates a 1d array that fulfills cond
    #val_new_obs = val_new_obs.flatten(); #makes the 2d into a 1d array
    #val_new_pred = val_new_pred.flatten(); #makes the 2d into a 1d array
    mat=val_new_obs[idx_lc]; # picks values from val_new_obs that fulfills cond
    mat1=val_new_pred[idx_lc]; # picks values from val_new_pred that fulfills cond
    

    #     ## indent 379-451
    # mat = np.ones((432,1));
    # mat1 = np.ones((432,1));
    # val_new_obs = np.ones((112,244,1));
    # val_new_pred = np.ones((112,244,1));
    # observation CDF
    # 33 percentile threshold for observation time series
    y1 = calc_plotting_position(mat);
    T1 = np.min(y1);
    T2 = np.max(y1);
    T3 = (T2-T1)/3;
    T4 = T1+T3;
    T5 = y1-T4;
    T6 = abs(T5);
    T7 = np.where(T6 == np.min(T6));
    below_no_obs = mat[T7];
    below_no_obs = below_no_obs.flatten();
    # 66 percentile threshold for observation time series
    T9 = T4 + T3;
    T10 = y1-T9;
    T11 = abs(T10);
    T12 = np.where(T11 == np.min(T11));
    above_no_obs = mat[T12];
    above_no_obs = above_no_obs.flatten();
    
    # prediction CDF
    # 33 percentile threshold for prediction time series
    y1 = calc_plotting_position(mat1);
    T1 = np.min(y1);
    T2 = np.max(y1);
    T3 = (T2-T1)/3;
    T4 = T1+T3;
    T5 = y1-T4;
    T6 = abs(T5);
    T7 = np.where( T6 == min(T6));
    below_no_pred = mat1[T7];
    below_no_pred = below_no_pred.flatten();
    # 66 percentile threshold for prediction time series
    T9 = T4+T3;
    T10 = y1-T9;
    T11 = abs(T10);
    T12 = np.where(T11 == np.min(T11));
    above_no_pred = mat1[T12];
    above_no_pred = above_no_pred.flatten();
    
    # populate categorical observation matrix
    for i in range((np.shape(fire_obs_ini_cate))[0]):
        for j in range((np.shape(fire_obs_ini_cate))[1]):
            if ((lc1[i][j] == lctypemat[lc_co]) & (val_new_obs[i][j][0] < below_no_obs).all()):
                val_new_obs[i][j] = -1;
            elif (lc1[i][j] == lctypemat[lc_co]) & (val_new_obs[i][j][0] > above_no_obs).all():
                val_new_obs[i][j]=1;
            elif (lc1[i][j] == lctypemat[lc_co]) & (val_new_obs[i][j][0] >= below_no_obs).all() & (val_new_obs[i][j] <= above_no_obs).all():
                val_new_obs[i][j] = 0;
            
    # populate categorical prediction matrix
    for i in range((np.shape(fire_pred_ini_cate))[0]):
        for j in range((np.shape(fire_pred_ini_cate))[1]):
            if (lc1[i][j] == lctypemat[lc_co]) & (val_new_pred[i][j][0] < below_no_pred).all():
                val_new_pred[i][j] = -1;
            elif (lc1[i][j]==lctypemat[lc_co]) & (val_new_pred[i][j][0] > above_no_pred).all():
                val_new_pred[i][j]=1;
            elif (lc1[i][j] == lctypemat[lc_co]) & (val_new_pred[i][j][0] >= below_no_pred).all() & (val_new_pred[i][j] <= above_no_pred).all():
                val_new_pred[i][j] = 0;
                  
    
    
    # categorical prediction and observation final matrices
    val_new_obs_tot_2 = np.empty((firemon_tot_sizeX,firemon_tot_sizeY));
    val_new_pred_tot_2 = np.empty((firemon_tot_sizeX,firemon_tot_sizeY));
    a = val_new_obs_tot_2[:][:];
    aa = val_new_obs;
    val_new_obs_tot_2 = val_new_obs;
    val_new_pred_tot_2 = val_new_pred;





# FIG 6 abd 7 of the paper for aug 2013

A = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
# August for title of the plot
figco = 8;

month_o_year = np.arange(0, (np.shape(val_new_pred_tot_1))[2], 11);
year = 11;

#month to graph histograms
mo = month_o_year[year] + 7;







# plot probabilities of observations
## figure(?)
#val = val_new_obs_tot_1[:][:][mo];
val_split = np.dsplit(val_new_obs_tot_1,132);
val = val_split[mo-1];
val = val.reshape((112,244))
# exclude LC types out of the scope of the study
for i in range(112):
    for j in range(244):
        if (lc1[i][k] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
            val[i][j] = float("NaN");

# x = 222;
# c = np.array([221]);

# while x != 243:
#     c = np.append(c,x);
#     x+=1;
# val = np.delete(val,c,axis=1);
val = np.rot90(val.T);
# val = np.rot90(val);
fig, (fig1) = plt.subplots(1, 1);
c = fig1.pcolor(val);
plt.xlabel(A[figco]);
plt.show(); 
# tx = np.arange(51.75, 23.875, -0.25);
# ty = np.arange(-126.75, -65.875, 0.25);
# mp.pyplot.pcolor(ty,tx,val);
# # axis image;
# # shading flat; 
# # caxis([0 1]);
# set(gcf,'PaperUnits','points');
# set(gcf,'Renderer','painters');
# # contin = load('coast');
# # hold on;plot(contin.long,contin.lat,'k-','linewidth',1);
# title(['Burned Area Observation Probability Residuals ' A{figco} ' 2013'])
# ylim([23.875 51.75 ])
# xlim([-126.75 -65.875])
# cmap=buildcmap('gwr');
# colormap(cmap)
# colorbar
# # states = shaperead('cb_2016_us_state_500k', 'UseGeoCoords', true);
# # geoshow(states,'FaceColor',[1,1,1],'facealpha',.3)



# # plot probabilities of prediction

# val_split = np.dsplit(val_new_pred_tot_1,132);
# val = val_split[mo-1];

# # exclude LC types out of the scope of the study
# for i in range(112):
#     for j in range(224):
#         if (lc1[i][k] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
#             val[i][j] = float("NaN");



# tx = np.arange(51.75, 23.875, -0.25);
# ty = np.arange(-126.75, -65.875, 0.25);
# pcolor(ty,tx,val)
# axis image;
# shading flat; 
# caxis([0 1]); 
# set(gcf,'PaperUnits','points');
# set(gcf,'Renderer','painters');
# # contin = load('CONUS_boundary');
# # hold on;plot(contin.long,contin.lat,'k-','linewidth',1);
# title(['Burned Area Prediction Probability Residuals ' A{figco} ' 2013'])
# ylim([23.875 51.75 ])
# xlim([-126.75 -65.875])
# # cmap=buildcmap('gwr');
# # colormap(cmap)
# colorbar
# # states = shaperead('cb_2016_us_state_500k', 'UseGeoCoords', true);
# # geoshow(states,'FaceColor',[1,1,1],'facealpha',.3)




# # plot categorical observations
# figure(3)
# val_split = np.dsplit(val_new_obs_tot_2,132);
# val = val_split[mo-1];

# # exclude LC types out of the scope of the study
# for i in range(112):
#     for j in range(224):
#         if (lc1[i][k] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
#             val[i][j] = float("NaN");


# tx = np.arange(51.75, 23.875, -0.25);
# ty = np.arange(-126.75, -65.875, 0.25);
# pcolor(ty,tx,val)
# axis image;
# shading flat; 
# caxis([-1 1]); 
# set(gcf,'PaperUnits','points');
# set(gcf,'Renderer','painters');
# # contin = load('coast');
# # hold on;plot(contin.long,contin.lat,'k-','linewidth',1);
# title(['Burned Area Observation Categorical ' A{figco} ' 2013'])
# ylim([23.875 51.75 ])
# xlim([-126.75 -65.875])
# cmap = [0 1 0;1 1 1;1 0 0 ];
# colormap(cmap)
# colorbar
# colorbar('YTick',[-1 0 1],'YTickLabel',{'below normal','normal','above normal'})
# # states = shaperead('cb_2016_us_state_500k', 'UseGeoCoords', true);
# # geoshow(states,'FaceColor',[1,1,1],'facealpha',.3) visualize maps





# # plot categorical predictions

# figure(4)
# for i in range(112):
#     for j in range(224):
#         if (lc1[i][k] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
#             val[i][j] = float("NaN");


# tx = np.arange(51.75, 23.875, -0.25);
# ty = np.arange(-126.75, -65.875, 0.25);
# pcolor(ty,tx,val)
# axis image;
# shading flat; 
# caxis([-1 1]);
# set(gcf,'PaperUnits','points');
# set(gcf,'Renderer','painters');
# # contin = load('coast');
# # hold on;plot(contin.long,contin.lat,'k-','linewidth',1);
# title(['Burned Area Prediction Categorical ' A{figco} ' 2013'])
# ylim([23.875 51.75 ])
# xlim([-126.75 -65.875])
# cmap = [0 1 0;1 1 1;1 0 0 ];
# colormap(cmap)
# colorbar
# colorbar('YTick',[-1 0 1],'YTickLabel',{'below normal','normal','above normal'})
# # states = shaperead('cb_2016_us_state_500k', 'UseGeoCoords', true);
# # geoshow(states,'FaceColor',[1,1,1],'facealpha',.3)












        
        
        
        
        
        
        
        
        
        