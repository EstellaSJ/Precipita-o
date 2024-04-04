# Instalando bibliotecas
!pip install netCDF4
!pip install cartopy
!pip install cmocean
!pip install gsw
!pip install webcolors

import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy.io import netcdf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import cartopy.feature
from datetime import datetime
from cartopy.mpl.patch import geos_to_path
import matplotlib.pyplot as plt
import numpy as np
import webcolors


# Obtendo os dados
imerge_mwprec_path = r'dados.nc' 
imerge_mwprec_nc = xr.open_mfdataset(imerge_mwprec_path)
mw_prec = imerge_mwprec_nc['precipitation'][:]
lat = imerge_mwprec_nc.lat.values
lon = imerge_mwprec_nc.lon.values
times = imerge_mwprec_nc.time.values

# Criando função colorbar 
def prec_wind_map(fig,ax,lon,lat,data,lvls = [25,28],lon_bound = [-47,-43],
            lon_shape = 21,lat_bound = [-25,-22], lat_shape = 13,
            name = "VHRR - 18/02/2023",dpi=100,rmse=False,mean=False,im=True,colorbar=True,intervalos = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]):
  land_50m = cfeature.NaturalEarthFeature('physical', 'land', '10m',edgecolor='black',linewidth=0.5, facecolor='None')
  cores_personalizadas = ['white', 'LightGreen', '#6fc276', 'MediumSeaGreen',  '#287c37', '#0b5509','#002d04',
                         'LightBlue', 'SkyBlue', 'DeepSkyBlue', 'DodgerBlue', 'RoyalBlue', 'Navy', 'Yellow', 'Gold', 'Orange', 'DarkOrange', 'Red', 'DarkRed']

  # Cria a escala de cores personalizada
  cmap = colors.ListedColormap(cores_personalizadas)
  norm = colors.BoundaryNorm(intervalos, cmap.N)
  ax.add_feature(land_50m)
  ax.coastlines(resolution='10m')
  ax.add_feature(cfeature.STATES)
  grad=[lon_bound[0],lon_bound[1],lat_bound[0],lat_bound[1]]
  ax.set_extent(grad, crs=ccrs.PlateCarree())
  g1 = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
  g1.top_labels = False
  g1.right_labels = False
  g1.xlabel_style = {'size': 12}
  g1.ylabel_style = {'size': 12}

  xticks = [-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50, 60, 70, 80]
  yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 55, 80]

  date_obj = datetime.strptime(str(datas[k+acumulacao]), "%Y-%m-%d %H:%M:%S")
  formatted_date = date_obj.strftime("%Y%m%d%HZ")
  ax.set_title("IMERG - " + formatted_date, fontsize=37, y=1.02)
  ax.add_feature(cfeature.STATES, linewidth=3, edgecolor='black')

  if im == False:
    return ax
  im = ax.contourf(lon,lat,data,levels=intervalos,zorder=-1,transform=ccrs.PlateCarree(),cmap=cmap,extend='both',norm=norm)
  props = dict(boxstyle='round', alpha=0)
  if mean:
    ax.text(0.1, 0.94, 'Média =', transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
    ax.text(0.22, 0.94, str(round(np.nanmean(data[data != np.inf]),3))+'mm',transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
  if rmse:
    ax.text(0.1, 0.84, 'RMSE =', transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
    ax.text(0.22, 0.84, str(round(np.nanmean(data**2),3)),transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
  return im,ax


# Interpolação
fig_p_v, ax = plt.subplots(7,3,figsize=(30,47),subplot_kw=dict(projection=ccrs.PlateCarree()))
datas = times
dadoint = mw_prec
k =134 #passo tempo da chuva
acumulacao = 2
cbar_int = 10
colorbar_range = []
for i in range(20):
  colorbar_range.append(int(cbar_int*i))
for i in range(7):
  for j in range(3):
    boolcolorbar = False
    dado = np.sum(dadoint[k:k+acumulacao,:,:],axis=0)*2
    #dado = np.where(dado <= 0.1, np.nan, dado)
    h = 1
    im,ax[i,j] = prec_wind_map(fig_p_v,ax[i,j],lon,lat,data=dado.T,lvls = [0,int(cbar_int*16)],
                  lon_bound = [-48,-43],lon_shape = len(lon),lat_bound = [-25,-22], lat_shape = len(lat),
                  name = "IMERG - "+str(datas[k+acumulacao]),dpi=100,intervalos = colorbar_range,colorbar=boolcolorbar)
    k = k+acumulacao

ponto_lon = -45.7661  
ponto_lat = -23.7550
for i in range(7):
    for j in range(3):
        ax[i, j].plot(ponto_lon, ponto_lat, 'ko', markersize=17, transform=ccrs.PlateCarree())

# Colorbar e títulos
cbar_ax = fig_p_v.add_axes([0.92, 0.115, 0.015, 0.76])
cbar = plt.colorbar(im, cax=cbar_ax, extendrect=True)
cbar.set_label(label='Precipitação acumulada (mm)', size=36, labelpad=20, position=(3.9, 0.0))
cbar.ax.yaxis.set_label_coords(1, 0.5)
cbar.ax.yaxis.set_label_coords(2.7, 0.5)
cbar.ax.tick_params(labelsize=36)

# Plotando e salvando
plt.show()
fig_p_v.savefig(r'figura.png',bbox_inches='tight')
