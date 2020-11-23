import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import utils

def getWeights(lats):

    """
    Calculates weights as cosine of latitude

    """

    wgts = np.cos(np.deg2rad(lats))
    return(wgts)

def getFcstDates(files):

    """
    Determines the dates of the initial conditions
    from SubX filename
    """

    dates_list=[]
    for f in files:
        fstr_1=f.split('.',1)[0].split('_')[-1]
        dates_list.append(pd.to_datetime(fstr_1,format='%Y%m%d'))

    return(pd.to_datetime(dates_list))


def makeGiorgiRegions(ds,ds_gmask,gregions,gn):

    """
    Calculates averaged values in specified Giorgi regions
    """
    
    # Assign the giorgi mask as a coordinate
    ds.coords['gmask'] = (('lat', 'lon'), 
                        ds_gmask['giorgimask'].squeeze())

    ds_rlist=[]
    for (ireg,reg) in zip(gn,gregions):

        ds=ds*getWeights(ds['lat'])
        ds_tmp=ds.where(ds['gmask']==ireg+1).mean(dim=['lat','lon'],
                                             skipna=True)
        ds_tmp=ds_tmp.rename({'pr':reg})
        ds_rlist.append(ds_tmp)
        
    ds_regs=xr.merge(ds_rlist)

    return ds_regs

def calcSkill(model,verif,var_list):

    """
    Calculates correlation as a function of lead time
    for given list of variables using verif and model.
    """

    # Match verif and hindcast ics
    verif_match=verif.sel(ic=model['ic'])

    # Calculate correlation for specificed regions
    corr=[]
    for v in var_list:
        corr.append((xr.corr(model[v],verif_match[v],
                     dim='ic')).to_dataset(name=v))
    corr_regions=xr.merge(corr)

    return corr_regions

def calcSkillSliding(model,verif,var_list,win_list):

    """
    Calculates correlation over a list of 
    averaging windows
    """

    # Loop over list of smoothing options
    skill_list=[]
    for ismooth in win_list:

        # Smooth the data
        model_smooth=model.rolling(time=ismooth, center=True).mean()
        verif_smooth=verif.rolling(time=ismooth, center=True).mean()

        # Calculate the correlations

        skill_list.append(calcSkill(model_smooth,
                                    verif_smooth,
                                    var_list))

    return skill_list

def plotSkill(corr,v,figname):

    """
    Plot the skill for each variable
    as a function of lead time
    """

    fig=plt.figure()

    for i,reg in enumerate(v):
        plt.subplot(3,1,i+1)
        plt.plot(corr[reg],'k')
        plt.ylim(-0.2,1.0)
        plt.title(reg)

    plt.tight_layout()

    plt.savefig('../figs/'+figname+'.png')


def getSubxFiles(path,model,group,vname,vlev,file_stub):
    
    files=path+group+'-'+model+'/'+\
          vname+'_'+vlev+'_'+group+\
          '-'+model+file_stub+'.nc'
    
    files=sorted(glob.glob(files))
   
    return files


def setCoords(ds,files):
    
    ds['ic']=utils.getFcstDates(files)
    ds['time']=np.arange(len(ds['time']))
    
    return ds
    

def reverseLats(ds):
    """
    Reverse lats if needed
    """
    
    if (ds['lat'][0]>ds['lat'][-1]):
        ds=ds.reindex(lat=list(reversed(ds['lat'])))
    
    return ds


def getGiorgiMask():

    path='/homes/kpegion/projects/finished/mappnmmepred/monthly/src/'
    file=path+'giorgi/giorgimask.nc'
    ds=xr.open_dataset(file).squeeze()
    
    return(ds)

def selCONUS(da):
    return(da.sel(lat=slice(25,59),lon=slice(200,300)))

def getSubxVerifFiles(path,dataset,vname,vlev,file_stub):
    
    files=path+vname+'_'+vlev+'_'+dataset+file_stub+'.nc'    
    files=sorted(glob.glob(files))

    return files

def getSubxVerifGiorgi(path,dataset,vname,vlev,
                       file_stub,ds_gmask,gregions,
                       gregnums):


    files=utils.getSubxVerifFiles(path,dataset,
                                  vname,vlev,
                                  file_stub)
    
    ds=xr.open_mfdataset(files,
                         combine='nested',
                         concat_dim='ic')  
     
    ds=utils.setCoords(ds,files)  
    
    ds_giorgi=utils.makeGiorgiRegions(ds,ds_gmask,
                                      gregions,gregnums)
    
    return ds_giorgi


def getSubXModelGiorgi(path,model,group,vname,vlev,
                       file_stub,gmask,
                       gregions,gregnums):
    

    files=utils.getSubxFiles(path,model,group,
                             vname,vlev,file_stub)  
    
    ds=xr.open_mfdataset(files,combine='nested',
                               concat_dim='ic')
    ds=utils.setCoords(ds,files)        
    ds=utils.reverseLats(ds)           
    ds_giorgi=utils.makeGiorgiRegions(ds,gmask,
                                      gregions,gregnums)
    
    return(ds_giorgi)

def plotSlidingCorrs(all_models,modelnames,
                     gregions,colors,slist,figname):
    
    fig=plt.figure(figsize=(8.5,11))

    for i,reg in enumerate(gregions):
        print(i,reg)
        plt.subplot(3,1,i+1)

        for iwin,(c,w) in enumerate(zip(colors,slist)):
        
            # Plot each model
            for imodel,m in enumerate(modelnames):
                plt.plot(all_models['time'],
                         all_models[reg][imodel,iwin,:].T,
                         color=c,linestyle='-',alpha=0.8)  
#        if (i==0):
#            plt.legend([str(x) for x in smoothlist],
#                        title='Window (days)')

            
        plt.ylim(-0.3,1.1)
        plt.grid(True)
        plt.xlabel('lead(days)')
        plt.ylabel('ACC')
        plt.title(reg)
        
    # Big title at the top        
    plt.suptitle('SubX Ensemble Mean Precipitation Skill; All ICs')

    plt.tight_layout()

    plt.savefig('../figs/'+figname)

    
def calcPerfectSkill(ds_emean,ds_emem):
    
    svar=ds_emean.var(dim='ic')
    diff=ds_emem-ds_emean
    nvar=(diff.std(dim='ens')).var(dim='ic')

    s=svar/nvar
    s_sq=s*s
    n=len(ds_emem['ens'])
    
    x1=s_sq+1
    x2=s_sq+(1.0/n)
    r=s_sq/np.sqrt(x1*x2)
    
    return r
                   
def calcPerfectSliding(emean,emem,var_list,win_list):

    # Drop missing ICs
    #emem=emem.dropna(dim='ic',how='all')
    #emean=emean.dropna(dim='ic',how='all')
    
    # Loop over list of smoothing options
    skill_list=[]
    for ismooth in win_list:

        # Smooth the data
        emean_smooth=emean.rolling(time=ismooth, center=True).mean()
        emem_smooth=emem.rolling(time=ismooth, center=True).mean()

        # Calculate the correlations

        skill_list.append(utils.calcPerfectSkill(emean_smooth,
                                                 emem_smooth))

    return skill_list

def initSubxDict():
    
    gem_dict=dict(group='ECCC',model='GEM',nens_hcst=4)
    fim_dict=dict(group='ESRL',model='FIMr1p1',nens_hcst=4)
    cfsv2_dict=dict(group='NCEP',model='CFSv2',nens_hcst=4)
    ccsm4_dict=dict(group='RSMAS',model='CCSM4',nens_hcst=4)
    gefs_dict=dict(group='EMC',model='GEFS',nens_hcst=11)
    geos_dict=dict(group='GMAO',model='GEOS_V2p1',nens_hcst=4)
    nespc_dict=dict(group='NRL',model='NESM',nens_hcst=1)
    cesm30l_dict=dict(group='CESM',model='30LCESM1',nens_hcst=10)
    cesm46l_dict=dict(group='CESM',model='46LCESM1',nens_hcst=10)
    
    subx_dict=[gem_dict,fim_dict,cfsv2_dict,ccsm4_dict,gefs_dict,
               geos_dict,nespc_dict,cesm30l_dict]
    
    return subx_dict
