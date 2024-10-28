#第三版，加入了并行运算功能
from astropy import time, units as u, stats
from astropy.io import fits,ascii
from astropy.table import Table,vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
import astrometry
from astroquery.xmatch import XMatch
from datetime import datetime,timedelta
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from photutils.background import Background2D, MedianBackground
import pandas as pd
from PIL import Image
import re
import sep
import subprocess
# import stdpipe.templates
import time
import warnings
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import sharedmem
import sys
from collections import Counter

warnings.filterwarnings('ignore')
def is_date_format(folder_name):
        # 使用正则表达式检查文件夹名称是否为日期格式 yyyy-mm-dd
        pattern = re.compile(r'^2024-\d{2}-\d{2}$')
        return pattern.match(folder_name) is not None
def read_date_folders(rawdir):
        # 获取基础路径下的所有文件夹
        all_folders = [folder for folder in os.listdir(rawdir) if os.path.isdir(os.path.join(rawdir, folder))]
        
        # 过滤出符合日期格式的文件夹
        date_folders = [folder for folder in all_folders if is_date_format(folder)]

        raw_date_folders=[f'{rawdir}/{date_folder}' for date_folder in date_folders]

        print('原文件存放在日期文件夹:\n')
        [print(raw_date_folder) for raw_date_folder in raw_date_folders]
        
        return raw_date_folders
#定义一个显示图像的函数
def showimg(filename,fraction=1.,return_cut=False):

    nx,ny=fits.getheader(filename)['NAXIS1'],fits.getheader(filename)['NAXIS2']
    x_center,y_center=nx/2,ny/2
    x_left,x_right=x_center*(1.-fraction),x_center*(1+fraction)
    y_top,y_bottom=y_center*(1.-fraction),y_center*(1+fraction)
    data=fits.getdata(filename)
    data_cut=data[int(y_top):int(y_bottom),int(x_left):int(x_right)]
    # 对比度
    vmin=np.percentile(data_cut[np.isnan(data_cut)==False],1)
    vmax=np.percentile(data_cut[np.isnan(data_cut)==False],99)
    plt.figure(figsize=(8,6))
    plt.imshow(data_cut,cmap='gray',vmin=vmin,vmax=vmax)
    plt.title(f'{filename}\n{fraction*100}%',fontsize=10)
    plt.axis('off')
    
    if return_cut:
        return data_cut

def fnlist_reshape(fnlist,run_cpu):
    
    n_cpu=mp.cpu_count()
    
    if run_cpu>n_cpu:
        print('run_cpu too larger than n_cpu, set run_cpu to n_cpu')
        run_cpu=n_cpu
        q,r=divmod(len(fnlist),run_cpu)
        listcut1=fnlist[:run_cpu*q]
        listcut2=fnlist[run_cpu*q:]
        listcut1=np.reshape(listcut1,(q,run_cpu))
    elif run_cpu<n_cpu:
        q,r=divmod(len(fnlist),run_cpu)
        listcut1=fnlist[:run_cpu*q]
        listcut2=fnlist[run_cpu*q:]
        listcut1=np.reshape(listcut1,(q,run_cpu))
    listcuts=list(listcut1)
    listcuts.append(listcut2)
    return listcuts
def overscan_cut(filename,x1=35,x2=3390,y1=30,y2=2570,shape=(2574, 3584)):
    file=fits.open(filename)
    data=file[0].data
    if data.shape==shape:
        data_cut=data[y1:y2,x1:x2]
        file[0].data=data_cut
        file.writeto(filename,overwrite=True)
#输入参数为本底文件名构成的列表，本底会保存在reddir下
def BiasCombine(biasfnlst,bias_file_path,opstep_path,swarp_cmd):

    ny,nx=fits.getval(biasfnlst[0],'NAXIS2'),fits.getval(biasfnlst[0],'NAXIS1')
    print(f'BIAS IMAGE SHAPE = {nx} * {ny}')

    cmd=swarp_cmd
    
    for biasfn in biasfnlst:
        cmd+=f' {biasfn}'

    cmd+=f' -c {opstep_path}/default.swarp'
    cmd+=f' -IMAGEOUT_NAME {bias_file_path} -WEIGHTOUT_NAME {opstep_path}/coadd.weight.fits -COMBINE Y -COMBINE_TYPE MEDIAN -RESAMPLE N -SUBTRACT_BACK N -DELETE_TMPFILES Y -CLIP_SIGMA 3.0'

    os.system(cmd)
#输入参数为平场文件名构成的列表，保存的平场文件名
def FlatCombine(flatfnlst,flat_file_path,bias_file_path,opstep_path,swarp_cmd):

    ny,nx=fits.getval(flatfnlst[0],'NAXIS2'),fits.getval(flatfnlst[0],'NAXIS1')
    print(f'FLAT IMAGE SHAPE = {nx} * {ny}')

    bias=fits.getdata(bias_file_path)

    flat_temp_name_list=[]

    for i in range(len(flatfnlst)):
        flat=fits.getdata(flatfnlst[i])
        hdr=fits.getheader(flatfnlst[i])
        flat=1.*flat-bias
        flat_median=np.nanmedian(flat)
        flat_temp=flat/flat_median
        flat_temp_name=flatfnlst[i].replace('.fits','_temp.fits')
        flat_temp_name_list.append(flat_temp_name)
        # hdr[f'flat{i+1}']=flatfnlst[i]
    # flat=np.float32(np.nanmedian(flat_cube,axis=0))
        fits.writeto(flat_temp_name,data=flat_temp,header=hdr,overwrite=True)

    cmd=swarp_cmd
    
    for flat_temp_name in flat_temp_name_list:
        cmd+=f' {flat_temp_name}'

    cmd+=f' -c {opstep_path}/default.swarp'
    cmd+=f' -IMAGEOUT_NAME {flat_file_path} -WEIGHTOUT_NAME {opstep_path}/coadd.weight.fits -COMBINE Y'
    cmd+=' -COMBINE_TYPE MEDIAN -RESAMPLE N -SUBTRACT_BACK N -DELETE_TMPFILES Y -CLIP_SIGMA 3.0'
    os.system(cmd)

    for flat_temp_name in flat_temp_name_list:
        os.system(f'rm {flat_temp_name}')

def check_upload(path):
    print('DATA UPLOADING !!! PLEASE WAIT !!!')
    while True:
        initial_size=os.path.getsize(path)
        time.sleep(1)
        current_size=os.path.getsize(path)
        if current_size-initial_size==0:
            break
        else:
            continue
#输入参数为科学图像文件名构成的列表，平场文件名
def CCDproc(lightfn,bias_file_path,flat_file_path,lightfnlst_band,band):
    
    i=np.where(np.array(lightfnlst_band)==lightfn)[0][0]+1
    
    lightfn_bf=lightfn.replace('rawdir','reddir')
    lightfn_bf=lightfn_bf.replace('.fits','_bf.fits')
    lightfn_bf=lightfn_bf.replace(' ','')
    
    if os.path.exists(lightfn_bf):
        print(f'PREPROCESSING ... {lightfn}')
        print(f'{band} BAND IMAGE {i}/{len(lightfnlst_band)} PREPROCESSED IMAGE IN : \n{lightfn_bf}')
    else:
        print(f'PREPROCESSING ... {lightfn}')
        ny,nx=fits.getval(lightfn,'NAXIS2'),fits.getval(lightfn,'NAXIS1')
        print(f'LIGHT IMAGE SHAPE = {nx} * {ny}')

        bias=fits.getdata(bias_file_path)
        flat=fits.getdata(flat_file_path)

        rawdata=fits.getdata(lightfn)
        hdr=fits.getheader(lightfn)
        bfdata=(1.*rawdata-bias)/flat
        bfdata=np.array(bfdata,dtype=np.float32)

        saturate=get_saturate(rawdata,bfdata)

        hdr['SATURATE']=saturate

        fits.writeto(lightfn_bf,data=bfdata,header=hdr,overwrite=True)
        print(f'{band} BAND IMAGE {i}/{len(lightfnlst_band)} PREPROCESSED IMAGE IN : \n{lightfn_bf}')

def tag(state):
    xopstep="""
                 ██╗  ██╗      ██████╗ ██████╗ ███████╗████████╗███████╗██████╗ 
                 ╚██╗██╔╝     ██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
                  ╚███╔╝█████╗██║   ██║██████╔╝███████╗   ██║   █████╗  ██████╔╝
                  ██╔██╗╚════╝██║   ██║██╔═══╝ ╚════██║   ██║   ██╔══╝  ██╔═══╝ 
                 ██╔╝ ██╗     ╚██████╔╝██║     ███████║   ██║   ███████╗██║     
                 ╚═╝  ╚═╝      ╚═════╝ ╚═╝     ╚══════╝   ╚═╝   ╚══════╝╚═╝     
    """
    print(xopstep)
    print('{0:-^96}'.format('Xinglong-Observatory Popular Science TElescope Pipeline'))
    print('{0:-^96}'.format('Version 3.0.0'))
    print('{0:-^96}'.format('Supported by XunhaoChen,YimingMao'))
    print('{0:-^96}'.format(state))
def sex(bffn,catfn,opstep_path,thres):
    order=f'source-extractor {bffn} -CATALOG_NAME {catfn} -DETECT_THRESH {thres} -ANALYSIS_THRESH {thres}'
    order+=f' -c {opstep_path}/default.sex -PARAMETERS_NAME {opstep_path}/default.param'
    os.system(order)

def sextractor(newfn,opstep_path,sextractor_cmd,pixel_scale,newfnlst):
    
    catfn = newfn.replace('.new', '.cat')
    i=np.where(np.array(newfnlst)==newfn)[0][0]+1
    
    # 检查是否存在所需的文件
    if os.path.exists(catfn):
        print(f'PHOTOMETRYING ... {newfn}')
        print(f"PHOTOMETRY {i}/{len(newfnlst)} CATALOG SAVE IN : \n{catfn}")
    else:
        print(f'PHOTOMETRYING ... {newfn}')
        cmd=f'{sextractor_cmd} {newfn} -c {opstep_path}/default.sex -PARAMETERS_NAME {opstep_path}/default.param '
        cmd+=f'-PIXEL_SCALE {pixel_scale} -CATALOG_NAME {catfn}'
        os.system(cmd)
        print(f"PHOTOMETRY {i}/{len(newfnlst)} CATALOG SAVE IN : \n{catfn}")

def cut_image(filename,xregion,yregion):

    data=fits.getdata(filename)
    hdr=fits.getheader(filename)
    ny,nx=data.shape

    q_x,r_x=divmod(nx,xregion)
    x_position=np.linspace(0,(xregion-1)*q_x,xregion)
    x_position=np.append(x_position,nx)
    x_position=np.array(x_position,dtype=int)
    q_y,r_y=divmod(ny,yregion)
    y_position=np.linspace(0,(yregion-1)*q_y,yregion)
    y_position=np.append(y_position,ny)
    y_position=np.array(y_position,dtype=int)

    for i in range(xregion):
        for j in range(yregion):
            data_cut=data[y_position[j]:y_position[j+1],x_position[i]:x_position[i+1]]
            fits.writeto(f'{filename[:-5]}_{i+1}_{j+1}.fits',data=data_cut,header=hdr,overwrite=True)

def rm_cut_image(filename,xregion,yregion):
    for i in range(xregion):
        for j in range(yregion):
            os.system(f'rm {filename[:-5]}_{i+1}_{j+1}.fits')

def stack_image(filename,xregion,yregion):
    
    data_stack=np.full(yregion*xregion,fill_value=None)
    data_stack=np.reshape(data_stack,(yregion,xregion))

    for i in range(xregion):
        for j in range(yregion):
            dataij=fits.getdata(f'{filename[:-5]}_{i+1}_{j+1}.fits')
            
            data_stack[j,i]=dataij
    hdrij=fits.getheader(f'{filename[:-5]}_{i+1}_{j+1}.fits')
    
    data_stack_row=[]
    for j in range(yregion):
        data_stack_rowj=np.hstack((data_stack[j]))
        data_stack_row.append(data_stack_rowj)

    data_stack=np.vstack(data_stack_row)
    
    fits.writeto(filename,data=data_stack,header=hdrij)

def sextractor_wcs(wcsfn,opstep_path,sextractor_cmd,pixel_scale,wcsfnlst):
    
    catfn = wcsfn.replace('.fits', '.cat')
    catfn=catfn.replace('WCS_IMAGE', 'WCS_CATALOG')
    i=np.where(np.array(wcsfnlst)==wcsfn)[0][0]+1
    
    # 检查是否存在所需的文件
    if os.path.exists(catfn):
        print(f'PHOTOMETRYING ... {wcsfn}')
        print(f"PHOTOMETRY {i}/{len(wcsfnlst)} CATALOG SAVE IN : \n{catfn}")
    else:
        print(f'PHOTOMETRYING ... {wcsfn}')
        cmd=f'{sextractor_cmd} {wcsfn} -c {opstep_path}/default.sex -PARAMETERS_NAME {opstep_path}/default.param '
        cmd+=f'-PIXEL_SCALE {pixel_scale} -CATALOG_NAME {catfn}'
        os.system(cmd)
        print(f"PHOTOMETRY {i}/{len(wcsfnlst)} CATALOG SAVE IN : \n{catfn}")

def imagecombine(wcsfnlst,red_date_folder,date,object,band,i,n,opstep_path):

    comfn = f'{red_date_folder}/COM_IMAGE/{date}_{object}_{band}_com_{len(wcsfnlst)}.fits'
    # 判断第一个文件列表是否存在
    if os.path.exists(comfn):
        print(f'IMAGECOMBINING ... {comfn}')
        print(f'IMAGECOMBINE {i}/{n} COMBINED IMAGE SAVE IN : \n{comfn}')
    else:
        os.system(f'rm {red_date_folder}/COM_IMAGE/{date}_{object}_{band}_com*.fits')

        print(f'IMAGECOMBINING ... {comfn}')
        if len(wcsfnlst) > 1:
            try:
                cmd=f"SWarp {red_date_folder}/WCS_IMAGE/*{object}*_{band}_*wcs.fits"
                cmd+=f' -c {opstep_path}/default.swarp -IMAGEOUT_NAME {comfn} -WEIGHTOUT_NAME {opstep_path}/coadd.weight.fits'
                cmd+=f' -COMBINE Y -COMBINE_TYPE MEDIAN -RESAMPLE Y -SUBTRACT_BACK N -DELETE_TMPFILES Y -CLIP_SIGMA 3.0' 
                os.system(cmd)
                os.remove(f'{opstep_path}/coadd.weight.fits')
                print(f'IMAGECOMBINE {i}/{n} COMBINED IMAGE SAVE IN : \n{comfn}')
            except:
                print(f'COMBINE FAILURE : \n{wcsfnlst}')
        if len(wcsfnlst) == 1:
            os.system(f'cp {wcsfnlst[0]} {comfn}')
            print(f'ONLY ONE {object} {band} IMAGE, COPY THAT AS COMBINED IMAGE : \n{comfn}')

        if len(wcsfnlst) == 0:
            print(f'NO {object} {band} BAND IMAGE EXISTS')
            
def calibrate(catfn,catfnlst):
    
    i=np.where(np.array(catfnlst)==catfn)[0][0]+1

    if 'ZERO_POINT' in fits.getheader(catfn):
        print(f"CALIBRATE {i}/{len(catfnlst)} WCS CATALOG SAVE IN : \n{catfn}")
    else:
        print(f'CALIBRATING... {catfn}')
        cat_data=fits.getdata(catfn,2)
        table=Table(cat_data)

        table_cross_match = XMatch.query(cat1=table, 
                            cat2='vizier:I/355/gaiadr3',
                            max_distance=5 * u.arcsec, colRA1='X_WORLD',colDec1='Y_WORLD',
                            colRA2='RAJ2000',colDec2='DEJ2000')

        header=fits.getdata(catfn,1)[0][0]

        cat_hdr=''
        for line in header:
            cat_hdr+=line+'\n'
        cat_hdr=fits.Header.fromstring(cat_hdr,'\n')

        filter=cat_hdr['FILTER']

        if filter=='R':
            gaia_band='RPmag'
        if filter=='G':
            gaia_band='Gmag'
        if filter=='B':
            gaia_band='BPmag'
        if filter=='L':
            gaia_band='Gmag'
        if filter=='i':
            gaia_band='Gmag'
        if filter=='g':
            gaia_band='Gmag'

        x=np.array(table_cross_match[gaia_band])
        y=np.array(table_cross_match['MAG_AUTO'])

        nan_index=np.array(list(set(np.append(np.where(np.isnan(x))[0],np.where(np.isnan(y))[0]))))
        if len(nan_index)!=0:
            x=np.delete(x,nan_index)
            y=np.delete(y,nan_index)
        sigma_clip=stats.sigma_clip(x-y,1).mask
        x=np.delete(x,np.where(sigma_clip)[0])
        y=np.delete(y,np.where(sigma_clip)[0])
        
        cali_grad,cali_zp=np.polyfit(x, y, 1)
        cali_grad,cali_zp=cali_grad,cali_zp
        sigma=np.std(y-cali_zp-x)
        cat_hdr['INI_SOURCES']=(len(table),'Initial sources in catalog')
        cat_hdr['MATCHED_SOURCES']=(len(table_cross_match),'Matched sources with GAIA DR3')
        cat_hdr['CALIBRATION']=(gaia_band,'Calibrate band in GAIA DR3')
        cat_hdr['CALI_GRAD']=(round(cali_grad,5),'Slope in calibration with GAIA DR3')
        cat_hdr['ZERO_POINT']=(round(cali_zp,3),'Zero point in calibration with GAIA DR3')
        cat_hdr['PHOT_ACCURACY']=(round(sigma,5),'Photometric accuracy in calibration with GAIA DR3')
        
        table_cross_match['MAG_CALI']=table_cross_match['MAG_AUTO']-cali_zp
        table_cross_match['MAGERR_CALI']=table_cross_match['MAG_CALI']-table_cross_match[gaia_band]

        cat_hl=fits.HDUList([
            fits.PrimaryHDU(header=cat_hdr),
            fits.BinTableHDU(data=table_cross_match)
        ])
        cat_hl.writeto(catfn,overwrite=True)
        print(f"CALIBRATE {i}/{len(catfnlst)} WCS CATALOG SAVE IN : \n{catfn}")

def check_size(filename,size):
    fnsize=os.stat(filename).st_size
    return fnsize==size

def solve_field(bffn,pixel_scale,solve_field_cmd,red_date_folder,bffnlst):

    newfn = bffn.replace('.fits', '.new')
    i=np.where(np.array(bffnlst)==bffn)[0][0]+1
    
    # 检查是否存在所需的文件
    if os.path.exists(newfn):
        print(f'ASTROMETRYING ... {bffn}')
        print(f"ASTROMETRY {i}/{len(bffnlst)} NEW IMAGE SAVE IN : \n{newfn}")
    elif os.path.exists(f'{red_date_folder}/FAILURE/{os.path.basename(bffn)}'):
        print(f'solve-field FAILURE : {bffn}')
    else:
        print(f'ASTROMETRYING ... {bffn}')
        hdr=fits.getheader(bffn)
        ra,dec=hdr['OBJCTRA'],hdr['OBJCTDEC']
        c = SkyCoord(f'{ra} {dec}', unit=(u.hourangle, u.deg))
        ra,dec=round(c.ra.deg,3),round(c.dec.deg,3)

        scale_low=pixel_scale-0.1
        scale_high=pixel_scale+0.1

        cmd=f'{solve_field_cmd} --resort --ra {ra} --dec {dec} --radius 2.5 --scale-units arcsecperpix --scale-low {scale_low} --scale-high {scale_high}'
        cmd+=f" --no-plots -O  --downsample 4 -M none -R none -S none -B none -U none {bffn}"

        os.system(cmd)

        try:
            axy=bffn.replace('.fits','.axy')
            wcs=bffn.replace('.fits','.wcs')
        
            os.remove(axy)
            os.remove(wcs)
        except:
            print(f'solve-field FAILURE : {bffn}')
            os.makedirs(f'{red_date_folder}/FAILURE',exist_ok=True)
            os.system(f'mv {bffn} {red_date_folder}/FAILURE')
            
        print(f"ASTROMETRY {i}/{len(bffnlst)} NEW IMAGE SAVE IN : \n{newfn}")

def scamp(catfn,opstep_path,scamp_cmd,catfnlst):

    headfn=catfn.replace('.cat','.head')
    ldacfn=catfn.replace('.cat', '.ldac')
    i=np.where(np.array(catfnlst)==catfn)[0][0]+1

    if os.path.exists(headfn):
        print(f"SCAMP {i}/{len(catfnlst)} HEADER SAVE IN : \n{headfn}")
    else:
        cmd=f'{scamp_cmd} {catfn} -c {opstep_path}/default.scamp -ASTREF_CATALOG FILE -ASTREFCAT_NAME {ldacfn} -CHECKPLOT_DEV NULL'
        os.system(cmd)
        print(f"SCAMP {i}/{len(catfnlst)} HEADER SAVE IN : \n{headfn}")

def astrometry_solve(catfn,wcsfn,opstep_path,pixel_scale):

    bffn=catfn.replace('CATALOG','LIGHT')
    bffn=bffn.replace('_cat.fits','.fits')
    data=fits.getdata(bffn)
    hdr=fits.getheader(bffn)

    ra,dec=hdr['OBJCTRA'],hdr['OBJCTDEC']
    c = SkyCoord(f'{ra} {dec}', unit=(u.hourangle, u.deg))
    ra,dec=c.ra.deg,c.dec.deg

    print(f'OBJECT RA,DEC={round(ra,3),round(dec,3)}')

    solver = astrometry.Solver(
        astrometry.series_4100.index_files(
            cache_directory=f"{opstep_path}/astrometry_cache",
            scales={12,13},
        )
    )

    cat=fits.open(catfn)
    cat_data=cat[2].data
    cat_data=cat_data[np.argsort(cat_data['MAGERR_AUTO'])]

    #应对天气不好,检测不出星
    if len(cat_data) < 100:
        print(f'SOLVE FAILURE : NO ENOUGH SOURCES IN {catfn}')
    else:
        stars=[[cat_data['XWIN_IMAGE'][i],cat_data['YWIN_IMAGE'][i]] for i in range(100)]
        solution = solver.solve(
            stars=stars,
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=pixel_scale-0.1,
                upper_arcsec_per_pixel=pixel_scale+0.1,
            ),
            position_hint=astrometry.PositionHint(
                ra_deg=ra,
                dec_deg=dec,
                radius_deg=2.5,
            ),
            solution_parameters=astrometry.SolutionParameters(),
        )

        if solution.has_match():

            print(f"{solution.best_match().center_ra_deg=}")
            print(f"{solution.best_match().center_dec_deg=}")
            print(f"{solution.best_match().scale_arcsec_per_pixel=}")
            wcs=solution.best_match().astropy_wcs()

            wcs_hdr=wcs.to_header()

            fits.writeto(wcsfn,header=hdr+wcs_hdr,data=data,overwrite=True)
        else:
            print(f'SOLVE FAILURE : {catfn}')
            
def run_hotpants(sci_img,hotpants_cmd,template,red_date_folder,rpjfnlst):
    
    i=np.where(np.array(rpjfnlst)==sci_img)[0][0]+1
    basename=os.path.basename(sci_img)
    obj=basename.split('_')[1]
    filter=basename.split('_')[2]
    temp_img=f"{template}/{obj}_{filter}_temimg.fits"
    diff_img=f"{red_date_folder}/DIF_IMAGE/{basename.replace('.fits','_diff.fits')}"
    sci_hdr=fits.getheader(sci_img)
    temp_hdr=fits.getheader(temp_img)
    sci_str=round(sci_hdr['SATURATE']*0.9)
    temp_str=round(temp_hdr['SATURATE']*0.9)
    sci_g=sci_hdr['GAIN']
    temp_g=temp_hdr['GAIN']
    if os.path.exists(diff_img):
        print(f'IMAGESUBTRATING ... {sci_img} SUBTRACTED {temp_img}')
        print(f'IMAGESUBTRATION {i}/{len(rpjfnlst)} : \n{diff_img} FROM {sci_img} SUBTRACTED {temp_img}')
    else:
        os.system(f"rm {red_date_folder}/DIF_IMAGE/{basename.replace('rpj.fits','')[:-2]}*rpj_diff.fits")
        print(f'IMAGESUBTRATING ... {sci_img} SUBTRACTED {temp_img}')
        order=f'{hotpants_cmd} -inim {sci_img} -tmplim {temp_img} -outim {diff_img} -n t -iu {sci_str} -tu {temp_str}'
        order+=f' -tl 1 -il 1 -ig {sci_g} -tg {temp_g} -iuk {sci_str} -tuk {temp_str}'
        os.system(order)
        print(f'IMAGESUBTRATION {i}/{len(rpjfnlst)} : \n{diff_img} FROM {sci_img} SUBTRACTED {temp_img}')

def extract_date(text):
    pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None
    
def cut_ldac(catfn,catfnlst,gaia_ra,gaia_dec,gaia_mag):

    ldacfn=catfn.replace('.cat', '.ldac')
    newfn=catfn.replace('.cat', '.new')
    i=np.where(np.array(catfnlst)==catfn)[0][0]+1

    # 检查是否存在所需的文件
    if os.path.exists(ldacfn):
        print(f"LDAC {i}/{len(catfnlst)} CATALOG SAVE IN : \n{ldacfn}")
    else:
        new=fits.open(newfn)
        wcs=WCS(new[0].header)
        img_shape=np.shape(new[0].data)
        xys=wcs.wcs_pix2world([[0,0],[img_shape[1],img_shape[0]]],0)
        minra=np.min([xys[0][0],xys[1][0]]);maxra=np.max([xys[0][0],xys[1][0]])
        mindec=np.min([xys[0][1],xys[1][1]]);maxdec=np.max([xys[0][1],xys[1][1]])
        # cat=fits.getdata(catfn,2)
        # x_world,y_world=cat['X_WORLD'],cat['Y_WORLD']
        # minra=np.min(x_world);maxra=np.max(x_world)
        # mindec=np.min(y_world);maxdec=np.max(y_world)
        # minra=minra-0.2;maxra=maxra+0.2
        # mindec=mindec-0.2;maxdec=maxdec+0.2
        if mindec<-90:
            mindec=-90
        if maxdec>90:
            maxdec=90
        if minra<0:
            minra=360-minra
        if maxra>360:
            maxra=maxra-360
        if minra<maxra and np.abs(minra-maxra)<200:
            bools=~((gaia_ra<minra)+(gaia_ra>maxra)+(gaia_dec<mindec)+(gaia_dec>maxdec))
        else:
            bools=~(~((gaia_ra<minra)+(gaia_ra>maxra))+(gaia_dec<mindec)+(gaia_dec>maxdec))
        need_ra=gaia_ra[bools]
        need_dec=gaia_dec[bools]
        need_mag=gaia_mag[bools]
        errs=np.zeros(len(need_ra))
        obsdate=2016.5*np.ones(len(need_ra))
        c1=fits.Column(name='X_WORLD',array=need_ra,format='D')
        c2=fits.Column(name='Y_WORLD',array=need_dec,format='D')
        c3=fits.Column(name='ERRA_WORLD',array=errs,format='E')
        c4=fits.Column(name='ERRB_WORLD',array=errs,format='E')
        c5=fits.Column(name='MAG',array=need_mag,format='E')
        c6=fits.Column(name='MAGERR',array=errs,format='E')
        c7=fits.Column(name='FLAGS',array=errs,format='I')
        c8=fits.Column(name='OBSDATE',array=obsdate,format='D')

        table_hdu=fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8],name='LDAC_OBJECTS')
        primary_hdu=fits.PrimaryHDU()
        hdu=fits.HDUList([primary_hdu,table_hdu])
        hdu.writeto(ldacfn,overwrite=True)
        print(f"LDAC {i}/{len(catfnlst)} CATALOG SAVE IN : \n{ldacfn}")

def make_obs_log(imgfnlst,red_date_folder):

    obs_log_dict=[]

    for imgfn in imgfnlst:
        hdr=fits.getheader(imgfn)
        exposure=hdr['EXPOSURE']
        objctra=hdr['OBJCTRA']
        objctdec=hdr['OBJCTDEC']
        object=hdr['OBJECT']
        filter=hdr['FILTER']

        obs_log_dict.append((exposure,objctra,objctdec,object,filter))
    obs_log_dict_set=set(obs_log_dict)

    obs_log=list(obs_log_dict_set)
    obs_log_table_inverse=Table(obs_log)
    
    obs_log_table=Table()

    obs_log_table['OBJECT']=list(obs_log_table_inverse[3])
    obs_log_table['RA']=list(obs_log_table_inverse[1])
    obs_log_table['DEC']=list(obs_log_table_inverse[2])
    obs_log_table['EXPOSURE']=list(obs_log_table_inverse[0])
    obs_log_table['FILTER']=list(obs_log_table_inverse[4])
    
    n_list=[]
    dirname=os.path.dirname(imgfnlst[0])
    
    for i in range(len(obs_log_table)):
        row=obs_log_table[i]
        object,filter,exposure=row['OBJECT'],row['FILTER'],row['EXPOSURE']
        n=len(glob.glob(f'{dirname}/{object}*{filter}*{exposure}*'))
        n_list.append(n)
        
    obs_log_table['N']=n_list
    obs_log_table=obs_log_table[np.argsort(obs_log_table['OBJECT'])]
    print(obs_log_table)

    ascii.write(obs_log_table,f"{red_date_folder}/obs_log_{red_date_folder.split('/')[-1]}.txt",format='csv',overwrite=True)

def combine_hdr_dat(fitsfn,fitsfnlst):
    headfn = fitsfn.replace('.fits', '.head')
    wcsimgfn=fitsfn.replace('LIGHT','WCS_IMAGE')
    wcsimgfn=wcsimgfn.replace('.fits','_wcs.fits')
    i=np.where(np.array(fitsfnlst)==fitsfn)[0][0]+1
    if not os.path.exists(wcsimgfn):
        head_ascii=open(headfn).readlines()
        hdr=''
        for i in range(3,17):
            line=head_ascii[i]
            hdr+=line
        hdr=fits.Header.fromstring(hdr,'\n')
        fitsfile=fits.open(fitsfn)
        fitsfile[0].header+=hdr
        fitsfile.writeto(wcsimgfn)
        print(f'COMBINING HEADER AND FITS FILES {i}/{len(fitsfnlst)}')
def get_saturate(rawdata,bfdata):
    # rawdata=fits.getdata(rawimgfn)
    # bfdata=fits.getdata(bfimgfn)
    staurate_postion=np.where(rawdata==65535)
    if len(staurate_postion[0])==0:
        saturate=65535
    else:
        bfdata_staurate=bfdata[staurate_postion]
        saturate=np.min(bfdata_staurate)

    return saturate
def bkg_sub(comfn,subfn):
    
    data=np.array(fits.getdata(comfn),dtype=np.float32)
    hdr=fits.getheader(comfn)
    bkg=sep.Background(data)
    data_sub=data-bkg
    fits.writeto(subfn,data=data_sub,header=hdr,overwrite=True)
def get_tem_xyr_50mpc(tem,opstep_path,pixel_scale):

    hdr_50mpc=fits.getheader(f'{opstep_path}/50mpccatalog.fits',1)
    data_50mpc=fits.getdata(f'{opstep_path}/50mpccatalog.fits',1)

    obj_50mpc,dist_50mpc,Bmag_50mpc=data_50mpc['objname'],data_50mpc['bestdist'],data_50mpc['Bmag']
    ra_50mpc,dec_50mpc,r_50mpc=data_50mpc['ra'],data_50mpc['dec'],data_50mpc['d25']*60
    r_50mpc=r_50mpc/pixel_scale #pixs
    
    hdr=fits.getheader(tem)
    w=WCS(hdr)

    sky=SkyCoord(frame='fk5', ra=ra_50mpc, dec=dec_50mpc, unit='deg')
    x,y=w.world_to_pixel(sky)

    index_xy=[]
    x_size,y_size=w.array_shape[1],w.array_shape[0]

    for i in range(len(x)):

        if np.isnan(x[i])==False and np.isnan(y[i])==False:
            if 0<x[i]<x_size and 0<y[i]<y_size:

                index_xy.append(i)
    
    x_pix,y_pix,r_pix=x[index_xy],y[index_xy],r_50mpc[index_xy]
    obj,dist,Bmag=obj_50mpc[index_xy],dist_50mpc[index_xy],Bmag_50mpc[index_xy]
    ra,dec=ra_50mpc[index_xy],dec_50mpc[index_xy]
    bmag=Bmag+5*np.log10(dist*1e5)

    names=('objname','ra','dec','dist_Mpc','Bmag','bmag','x_pix','y_pix','r_pix')
    dtype=(str,np.float64,np.float64,np.float16,np.float16,np.float16,np.float16,np.float16,np.float16)

    xyr_table=Table([obj,ra,dec,dist,Bmag,bmag,x_pix,y_pix,r_pix],names=names,dtype=dtype)
    xyr_table=xyr_table[np.argsort(xyr_table['bmag'])]

    return xyr_table
def cut_size(bound,size):
    if bound < 0:
        return 0
    if bound > size:
        return size
    else:
        return bound
def save_pdf(jpgfnlst,pdffn):
    index=np.linspace(1,len(jpgfnlst),len(jpgfnlst),dtype=int)
    jpgfnlst_srcs=[]

    for jpgfn in jpgfnlst:
        jpgfn_src=jpgfn.split('.')[1]
        jpgfnlst_srcs.append(jpgfn_src)
    print(f'PEAK VALUE : {jpgfnlst_srcs}')
    table=Table(data=[jpgfnlst_srcs,index],names=('peak','index'),dtype=(int,int))
    table=table[np.argsort(-abs(table['peak']))]
    
    if len(jpgfnlst_srcs)==0:
        with PdfPages(pdffn) as pdf:
            plt.figure(figsize=(8,6))
            plt.text(0.5,0.5,f'NO SOURCES DETECTED!\nin\n{pdffn}',size=30,
                    horizontalalignment='center',verticalalignment='center')
            plt.axis('off');
            pdf.savefig(bbox_inches = 'tight')
    else:
        with PdfPages(pdffn) as pdf:
            for i in table['index']:
                img = Image.open(jpgfnlst[i-1])
                plt.figure(dpi=300)
                plt.imshow(img)
                plt.axis('off');
                pdf.savefig(bbox_inches = 'tight')
        
        
def draw_cutout_50mpc(diffn,opstep_path,pixel_scale,template,diffnlst):
    
    j=np.where(np.array(diffnlst)==diffn)[0][0]+1
    basename=os.path.basename(diffn)
    obj=basename.split('_')[1]
    filter=basename.split('_')[2]
    temfn=f"{template}/{obj}_{filter}_temimg.fits"
    xyr_table=get_tem_xyr_50mpc(temfn,opstep_path,pixel_scale)
    scifn=diffn.replace('DIF_IMAGE','RPJ_IMAGE')
    scifn=scifn.replace('_diff.fits','.fits')
    pdffn=diffn.replace('DIF_IMAGE','PDF')
    pdffn=pdffn.replace('.fits','.pdf')

    if os.path.exists(pdffn):
        print(f'PDF PLOTTING ... {diffn}')
        print(f'PLOT PDF {i}/{len(diffnlst)} : \n{pdffn}')
    else:
        print(f'PDF PLOTTING ... {diffn}')
        
        tem_data=fits.getdata(temfn)
        sci_data=fits.getdata(scifn)
        diff_data=fits.getdata(diffn)

        bkg_estimator = MedianBackground()
        bkg_diff = Background2D(diff_data, (50,50), filter_size=(3, 3),
                bkg_estimator=bkg_estimator)
        diff_bkg=bkg_diff.background
        diff_data -= diff_bkg

        y_size,x_szie=tem_data.shape
        
        date=re.search('\d{4}-\d{2}-\d{2}',diffn).group()
        table_names=['DATE','OBJECT','RA','DEC','X_IMAGE','Y_IMAGE','SOURCES',"TEMPLATE_PATH",'SCIENCE_PATH','DIFFERENCE_PATH','PDF_PATH']
        table_dtype=['str', 'str', 'float', 'float', 'float', 'float','int', 'str', 'str', 'str', 'str']
        table=Table(names=table_names,
                    dtype=table_dtype)
        
        for i in range(len(xyr_table)):
        # for i in range(1):
            x=xyr_table['x_pix'][i]
            y=xyr_table['y_pix'][i]
            r=xyr_table['r_pix'][i]

            if np.isnan(r):
                r=int(np.nanmedian(xyr_table['r_pix']))

            obj=xyr_table['objname'][i]
            dist=xyr_table['dist_Mpc'][i]
            bmag=xyr_table['bmag'][i]
            ra=xyr_table['ra'][i]
            dec=xyr_table['dec'][i]

            c = SkyCoord(ra,dec, unit=u.deg)
            ra_hms=c.ra.hms
            ra_h=int(ra_hms.h); ra_m=int(ra_hms.m); ra_s=round(ra_hms.s,3)
            dec_dms=c.dec.dms
            dec_d=int(dec_dms.d); dec_m=int(dec_dms.m); dec_s=round(dec_dms.s,3)

            info=f'OBJNAME:{obj}\nDISTANCE:{dist} Mpc\nBMAG:{bmag}\nx={int(x)},y={int(y)},width={int(r)} pixel\nra={ra},dec={dec}'
            info+=f'\nhms: {ra_h}:{ra_m}:{ra_s},dms: {dec_d}:{dec_m}:{dec_s}'

            x_left,x_right,y_top,y_bottom=int(x-1.5*r),int(x+1.5*r),int(y-1.5*r),int(y+1.5*r)
            x_left,x_right,y_top,y_bottom=cut_size(x_left,x_szie),cut_size(x_right,x_szie),cut_size(y_top,y_size),cut_size(y_bottom,y_size)

            tem_data_cutout=tem_data[y_top:y_bottom,x_left:x_right]
            sci_data_cutout=sci_data[y_top:y_bottom,x_left:x_right]
            diff_data_cutout=diff_data[y_top:y_bottom,x_left:x_right]

            if np.mean(sci_data_cutout)>1e-29:

                fig,ax=plt.subplots(1,3,figsize=(15,5))

                fig.text(0.51,1.05,info,ha='center',va='center',fontsize=15)

                ax[0].imshow(tem_data_cutout,cmap='gray',vmin=np.percentile(tem_data_cutout,1),vmax=np.percentile(tem_data_cutout,99))
                ax[0].set_title('TEMPLATE')
                ax[0].axis('off')
                
                ax[1].imshow(sci_data_cutout,cmap='gray',vmin=np.percentile(sci_data_cutout,1),vmax=np.percentile(sci_data_cutout,99))
                ax[1].set_title('SCIENCE')
                ax[1].axis('off')

                daofind = DAOStarFinder(fwhm=5.0, threshold=3.*np.std(diff_data_cutout))  
                sources = daofind(diff_data_cutout) 

                ax[2].imshow(diff_data_cutout,cmap='gray',vmin=np.percentile(diff_data_cutout,1),vmax=np.percentile(diff_data_cutout,99))
                ax[2].axis('off')

                if sources is not None:
                    
                    for col in sources.colnames:  
                        if col not in ('id', 'npix'):
                            sources[col].info.format = '%.2f'  # for consistent table output
                    peak=int(np.nanmax(sources['peak']))
                    # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                    # apertures = CircularAperture(positions, r=5.0)
                    # apertures.plot(color='red', lw=3.)
                    # ax[2].set_title(f'DIFFERENCE : {len(sources)} sources detected')
                    ax[2].set_title(f'DIFFERENCE')
                    fig.savefig(f'{opstep_path}/{basename[:-5]}_{i}.{peak}.jpg',bbox_inches = 'tight')
                    
                    n=len(sources)
                    table_row=Table()
                    table_row['DATE']=[date]
                    table_row['OBJECT']=obj
                    table_row['RA']=ra
                    table_row['DEC']=dec
                    table_row['X_IMAGE']=x
                    table_row['Y_IMAGE']=y
                    table_row['SOURCES']=n
                    table_row['TEMPLATE_PATH']=temfn
                    table_row['SCIENCE_PATH']=scifn
                    table_row['DIFFERENCE_PATH']=diffn
                    table_row['PDF_PATH']=pdffn
                    table=vstack([table,table_row])

        jpgfnlst=glob.glob(f'{opstep_path}/{basename[:-5]}*.jpg')
        
        save_pdf(jpgfnlst,pdffn)

        print(f'PLOT PDF {j}/{len(diffnlst)} : \n{pdffn}')

        databasename=f'{opstep_path}/database.csv'
        database=ascii.read(databasename)
        database=vstack([database,table])
        database=Table(np.unique(database))
        ascii.write(database,databasename,overwrite=True,format='csv')

    databasename=f'{opstep_path}/database.csv'
    df=pd.read_csv(databasename)
    df=df.sort_values(by='DATE')
    df=df.reset_index(drop=True)
    df.index+=1
    df.to_csv('database.csv',index_label='INDEX')

def make_template(temfn):
        
    basename=os.path.basename(temfn)
    dirname=os.path.dirname(temfn)
    obj=basename.split('_')[1]
    filter=basename.split('_')[2]
    newname=f'{dirname}/{obj}_{filter}_temimg.fits'
    os.rename(temfn,newname)
    headname=f'{dirname}/{obj}_{filter}_tem.head'
    head=fits.getheader(newname)
    fits.writeto(headname,header=head,data=None,overwrite=True)
        
def reproject_swarp(comfn,template,temfnlst,red_date_folder,comfnlst,
                    swarp_cmd,opstep_path,pixel_scale):

    basename=os.path.basename(comfn)
    i=np.where(np.array(comfnlst)==comfn)[0][0]+1
    object=basename.split('_')[1]
    filter=basename.split('_')[2]
    temphead=f"{template}/{object}_{filter}_tem.head"
    tempimg=f"{template}/{object}_{filter}_temimg.fits"
    if temphead not in temfnlst:
        print(f'REPROJECT FAILURE : TEMPLATE FILE {temphead} NO EXISTS')
        print(f'COPY {comfn} AS TEMPLATE FILE')
        os.system(f'cp {comfn} {template}/')
        make_template(f'{template}/{basename}')
    if temphead in temfnlst:
        rpj_fits=f"{red_date_folder}/RPJ_IMAGE/{basename.replace('.fits','_rpj.fits')}"
        output_fits=temphead.replace('.head','.fits')
        input_fits=comfn
        imgshape=fits.getdata(tempimg).shape
        imgsize=f'{imgshape[1]},{imgshape[0]}'
        if os.path.exists(rpj_fits):
            print(f'REPROJECTING ... {comfn}')
            print(f'REPROJECT {i}/{len(comfnlst)} : \n{rpj_fits} FROM {comfn} REPROJECTED TO {temphead}')
        else:
            os.system(f"rm {red_date_folder}/RPJ_IMAGE/{basename.replace('.fits','')[:-2]}*rpj.fits")
            print(f'REPROJECTING ... {comfn}')                
            #输入的fits需要有对应的head文件
            #head文件是要被对齐的wcs头
            input_head=input_fits.replace('.fits','.head')
            output_head=output_fits.replace('.fits','.head')
            input_hdr=fits.getheader(input_fits)
            output_hdr=fits.getheader(output_head)
            ra,dec=output_hdr['CRVAL1'],output_hdr['CRVAL2']
            c = SkyCoord(ra,dec, unit=u.deg)
            ra_hms=c.ra.hms
            ra_h=int(ra_hms.h); ra_m=int(ra_hms.m); ra_s=round(ra_hms.s,3)
            dec_dms=c.dec.dms
            dec_d=int(dec_dms.d); dec_m=int(dec_dms.m); dec_s=round(dec_dms.s,3)
            fits.writeto(input_head,header=input_hdr,data=None)
            cmd=swarp_cmd
            cmd+=f' {input_fits} -c {opstep_path}/default.swarp -IMAGEOUT_NAME {output_fits} -WEIGHTOUT_NAME {opstep_path}/coadd.weight.fits'
            cmd+=' -COMBINE Y -RESAMPLE Y -SUBTRACT_BACK N -DELETE_TMPFILES Y -FSCALASTRO_TYPE VARIABLE'
            cmd+=f' -CENTER_TYPE MANUAL -CENTER {ra_h}:{ra_m}:{ra_s},{dec_d}:{dec_m}:{dec_s}'
            cmd+=f' -PIXELSCALE_TYPE MANUAL -IMAGE_SIZE {imgsize} -PIXEL_SCALE {pixel_scale}'
            print(cmd)
            os.system(cmd)
            os.system(f'mv {output_fits} {rpj_fits}')
            os.remove(input_head)
            print(f'REPROJECT {i}/{len(comfnlst)} : \n{rpj_fits} FROM {comfn} REPROJECTED TO {temphead}')

def preprocessing(raw_date_folder,bands,opstep_path,pixel_scale,
                  solve_field_cmd,sextractor_cmd,scamp_cmd,swarp_cmd,
                  run_cpu,xregion,yregion):

    red_date_folder=raw_date_folder.replace('raw', 'red')

    print(f'     DEALING WITH      : {raw_date_folder}')
    print(f'OUTPUT WILL BE SAVE IN : {red_date_folder}\n')
    print('CREATING FOLDERS : LIGHT, CATALOG, WCS_IMAGE, LIGHT_COM')

    os.makedirs(red_date_folder, exist_ok=True)
    os.makedirs(f'{red_date_folder}/LIGHT', exist_ok=True)
    os.makedirs(f'{red_date_folder}/WCS_CATALOG', exist_ok=True)
    os.makedirs(f'{red_date_folder}/WCS_IMAGE', exist_ok=True)
    os.makedirs(f'{red_date_folder}/COM_IMAGE', exist_ok=True)
    
    print('{0:-^96}'.format('BEGIN PREPROCESS PROGRAM'))
    start = time.perf_counter()
    # 合并本底
    #读取rawdir中的文件夹，定义本底、平场、科学图像的列表变量
    biasfnlst=glob.glob(f'{raw_date_folder}/BIAS/*')
    print(f'\nBIAS : {len(biasfnlst)} FRAMES')
    bias_file_path=f'{red_date_folder}/bias.fits'
    if os.path.exists(bias_file_path):
        print(f"BIAS COMBINE IN : \n{bias_file_path}")
    else: 
        print(f'BIAS COMBINING ...')
        BiasCombine(biasfnlst,bias_file_path,opstep_path,swarp_cmd)
        print(f'BIAS COMBINE IN : \n{bias_file_path}')
        # for bias in biasfnlst:
        #     cut_image(bias,xregion,yregion)
        # pool=mp.Pool(processes=xregion*yregion)
        # for i in range(xregion):
        #     for j in range(yregion):
        #         imglist=glob.glob(f'{raw_date_folder}/BIAS/*_{i+1}_{j+1}.fits')
        #         bias_path=f'{raw_date_folder}/BIAS/bias_{i+1}_{j+1}.fits'
        #         pool.apply_async(BiasCombine,args=(imglist,bias_path,opstep_path,swarp_cmd))
        # pool.close()
        # pool.join()
        # stack_image(f'{raw_date_folder}/BIAS/bias.fits',xregion,yregion)
        # rm_cut_image(f'{raw_date_folder}/BIAS/bias.fits',xregion,yregion)
        # for bias in biasfnlst:
        #     rm_cut_image(bias,xregion,yregion)
        # os.system(f'mv {raw_date_folder}/BIAS/bias.fits {bias_file_path}')
        # print(f'BIAS COMBINE IN : \n{bias_file_path}')
    end = time.perf_counter()
    runTime = end - start
    print(f"BIAS COMBINE SPENT TIME : {round(runTime,3)} SEC\n")

    start = time.perf_counter()
    # 合并平场（不同滤光片的平场要分别合并）
    print('\nFLAT : ')
    pool = mp.Pool(processes=len(bands))
    for band in bands:
        flatfnlst_band=glob.glob(f'{raw_date_folder}/FLAT/*_{band}_*')
        print(f'{band} BAND HAS {len(flatfnlst_band)} FRAMES:')
        flat_file_path=f'{red_date_folder}/flat_{band}.fits'
        if os.path.exists(flat_file_path):
            print(f"{band} BAND FLAT COMBINE IN : \n{flat_file_path}")
        else:
            print(f'{band} BAND FLAT COMBINING ...')
            pool.apply_async(FlatCombine, args=(flatfnlst_band, flat_file_path, bias_file_path,opstep_path,swarp_cmd))
            # FlatCombine(flatfnlst_band, flat_file_path, bias_file_path,opstep_path,swarp_cmd)
            print(f"{band} BAND FLAT COMBINE IN : \n{flat_file_path}")
    pool.close()
    pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"FLAT COMBINE SPENT TIME : {round(runTime,3)} SEC\n")

    start = time.perf_counter()
    # 减本底、除平场
    # 检查每个列表中的文件是否有对应的 *_R_*bf.fits, *_G_*bf.fits, *_B_*bf.fits 文件
    # 检查是否所有文件都已经有对应的校正文件
    print('\nLIGHT : ')
    for band in bands:
        flat_file_path=f'{red_date_folder}/flat_{band}.fits'
        lightfnlst_band=glob.glob(f'{raw_date_folder}/LIGHT/*_{band}_*')
        print(f'{band} BAND HAS {len(lightfnlst_band)} FRAMES:')
        listcuts=fnlist_reshape(lightfnlst_band,run_cpu)
        for listcut in listcuts:
            if len(listcut)>0:
                pool = mp.Pool(processes=len(listcut))
                for fn in listcut:
                    pool.apply_async(CCDproc, args=(fn,bias_file_path,flat_file_path,lightfnlst_band,band))
                pool.close()
                pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"PREPROCESS SPENT TIME : {round(runTime,3)} SEC\n")

    #制作观测记录
    imgfnlst=glob.glob(f'{red_date_folder}/LIGHT/*bf.fits')
    print('OBSERVE LOG SAVED ...')
    if not os.path.exists(f"{red_date_folder}/obs_log_{red_date_folder.split('/')[-1]}.txt"):
        make_obs_log(imgfnlst,red_date_folder)

    # 创建一个集合来存储唯一的目标
    unique_targets = set()
    for image in imgfnlst:
        header = fits.getheader(image)
        target = header['OBJECT']
        target = target.replace(' ','')
        unique_targets.add(target)
    # 将集合转换为列表（如果需要进一步处理）
    unique_targets_list = sorted(list(unique_targets))
    date=raw_date_folder.split('/')[-1]

    print('{0:-^96}'.format('BEGIN ASTROMETRY PROGRAM'))
    start = time.perf_counter()
    # 解析
    bffnlst = glob.glob(f'{red_date_folder}/LIGHT/*bf.fits')
    listcuts=fnlist_reshape(bffnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(solve_field, args=(fn,pixel_scale,solve_field_cmd,red_date_folder,bffnlst))
            pool.close()
            pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"ASTROMETRY SPENT TIME : {round(runTime,3)} SEC\n")

    print('{0:-^96}'.format('BEGIN PHOTOMETRY PROGRAM'))
    start = time.perf_counter()
    # 测光
    newfnlst = glob.glob(f'{red_date_folder}/LIGHT/*bf.new')
    listcuts=fnlist_reshape(newfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(sextractor, args=(fn,opstep_path,sextractor_cmd,pixel_scale,newfnlst))
            pool.close()
            pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"PHOTOMETRY SPENT TIME : {round(runTime,3)} SEC\n")

    print('{0:-^96}'.format('BEGIN SCAMP PROGRAM'))
    start = time.perf_counter()
    # scamp
    catfnlst = glob.glob(f'{red_date_folder}/LIGHT/*bf.cat')
    ldacfnlst=glob.glob(f'{red_date_folder}/LIGHT/*bf.ldac')
    if len(catfnlst)!=len(ldacfnlst):
        gaia_ra=np.memmap(f'{opstep_path}/gaia/ra.npy',dtype='float64',mode='r')
        gaia_dec=np.memmap(f'{opstep_path}/gaia/dec.npy',dtype='float64',mode='r')
        gaia_mag=np.memmap(f'{opstep_path}/gaia/G.npy',dtype='float64',mode='r')
        gaia_ra_mp = sharedmem.copy(gaia_ra)
        gaia_dec_mp = sharedmem.copy(gaia_dec)
        gaia_mag_mp = sharedmem.copy(gaia_mag)

        listcuts=fnlist_reshape(catfnlst,run_cpu)
        for listcut in listcuts:
            if len(listcut)>0:
                pool = mp.Pool(processes=len(listcut))
                for fn in listcut:
                    pool.apply_async(cut_ldac, args=(fn,catfnlst,gaia_ra_mp,gaia_dec_mp,gaia_mag_mp))
                pool.close()
                pool.join()
    listcuts=fnlist_reshape(catfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(scamp, args=(fn,opstep_path,scamp_cmd,catfnlst))
            pool.close()
            pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"SCAMP SPENT TIME : {round(runTime,3)} SEC\n")

    # 合并head和fits
    print('COMBINING HEADER AND FITS FILES ...')
    fitsfnlst = glob.glob(f'{red_date_folder}/LIGHT/*bf.fits')
    listcuts=fnlist_reshape(fitsfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(combine_hdr_dat, args=(fn,fitsfnlst))
            pool.close()
            pool.join()

    print('{0:-^96}'.format('BEGIN PHOTOMETRY PROGRAM'))
    start = time.perf_counter()
    # 测光
    wcsfnlst = glob.glob(f'{red_date_folder}/WCS_IMAGE/*wcs.fits')
    listcuts=fnlist_reshape(wcsfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(sextractor_wcs, args=(fn,opstep_path,sextractor_cmd,pixel_scale,wcsfnlst))
            pool.close()
            pool.join()
    end = time.perf_counter()
    runTime = end - start
    print(f"PHOTOMETRY SPENT TIME : {round(runTime,3)} SEC\n")

    # 流量定标
    print('{0:-^96}'.format('BEGIN CALIBRATION PROGRAM'))
    start = time.perf_counter()
    catfnlst = glob.glob(f'{red_date_folder}/WCS_CATALOG/*wcs.cat')
    listcuts=fnlist_reshape(catfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(calibrate, args=(fn,catfnlst))
            pool.close()
            pool.join()            
    end = time.perf_counter()
    runTime = end - start
    print(f"CALIBRATION SPENT TIME : {round(runTime,3)} SEC\n")
    
    print('{0:-^96}'.format('BEGIN IMAGECOMBINE PROGRAM'))
    #图像合并
    start = time.perf_counter()
    i=0
    n=len(unique_targets_list)*len(bands)
    wcsfnlst_dict=[]
    for object in unique_targets_list:
        for band in bands:
            i+=1
            wcsfnlst=glob.glob(f'{red_date_folder}/WCS_IMAGE/*{object}*_{band}_*wcs.fits')    
            wcsfnlst_dict.append(wcsfnlst)
    q,r=divmod(len(wcsfnlst_dict),run_cpu)
    for i in range(q+1):
        listcuts=wcsfnlst_dict[i*run_cpu:(i+1)*run_cpu]
        if len(listcuts)>0:
            pool = mp.Pool(processes=len(listcuts))
            for listcut in listcuts:
                if len(listcut)>0:
                    listcut_band=fits.getheader(listcut[0])['FILTER']
                    listcut_obj=fits.getheader(listcut[0])['OBJECT']
                    listcut_obj=listcut_obj.replace(' ','')
                    pool.apply_async(imagecombine, args=(listcut,red_date_folder,date,listcut_obj,listcut_band,i,n,opstep_path))
            pool.close()
            pool.join()      
    end = time.perf_counter()
    runTime = end - start
    print(f"IMAGECOMBINE SPENT TIME : {round(runTime,3)} SEC\n")

def imagesubtraction(raw_date_folder,template,opstep_path,pixel_scale,
                     swarp_cmd,hotpants_cmd,run_cpu,xregion,yregion,folder,cppdf,today):
    
    #图像对齐
    start = time.perf_counter()
    red_date_folder=raw_date_folder.replace('raw', 'red')
    print('{0:-^96}'.format('BEGIN REPROJECT PROGRAM'))
    os.makedirs(f'{red_date_folder}/RPJ_IMAGE', exist_ok=True)
    comfnlst=glob.glob(f'{red_date_folder}/COM_IMAGE/*')
    temfnlst=glob.glob(f'{template}/*')
    listcuts=fnlist_reshape(comfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(reproject_swarp, args=(fn,template,temfnlst,red_date_folder,comfnlst,
                        swarp_cmd,opstep_path,pixel_scale))
            pool.close()
            pool.join()     
    end = time.perf_counter()
    runTime = end - start
    print(f"REPROJECT SPENT TIME : {round(runTime,3)} SEC\n")

    #图像相减
    start = time.perf_counter()
    print('{0:-^96}'.format('BEGIN IAMGESUBTRATION PROGRAM'))
    os.makedirs(f'{red_date_folder}/DIF_IMAGE', exist_ok=True)
    rpjfnlst=glob.glob(f'{red_date_folder}/RPJ_IMAGE/*')
    listcuts=fnlist_reshape(rpjfnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(run_hotpants, args=(fn,hotpants_cmd,template,red_date_folder,rpjfnlst))
            pool.close()
            pool.join()    
    end = time.perf_counter()
    runTime = end - start
    print(f"IMAGESUBTRATION SPENT TIME : {round(runTime,3)} SEC\n")
    diffnlst=glob.glob(f'{red_date_folder}/DIF_IMAGE/*')
    for diffn in diffnlst:
        if check_size(diffn,0):
            os.system(f'rm {diffn}')

    #画pdf  
    print('{0:-^96}'.format('BEGIN PLOTPDF PROGRAM'))
    os.makedirs(f'{red_date_folder}/PDF', exist_ok=True)
    start = time.perf_counter()
    diffnlst=glob.glob(f'{red_date_folder}/DIF_IMAGE/*')
    listcuts=fnlist_reshape(diffnlst,run_cpu)
    for listcut in listcuts:
        if len(listcut)>0:
            pool = mp.Pool(processes=len(listcut))
            for fn in listcut:
                pool.apply_async(draw_cutout_50mpc, args=(fn,opstep_path,pixel_scale,template,diffnlst))
            pool.close()
            pool.join()   
    end = time.perf_counter()
    runTime = end - start
    print(f"PLOTPDF SPENT TIME : {round(runTime,3)} SEC\n")
    
    pdffnlst=glob.glob(f'{red_date_folder}/PDF/*')
    for pdf in pdffnlst:
        if check_size(pdf,208):
            os.system(f'rm {pdf}')
    
    if cppdf==True:
        os.makedirs(f'/data/transient/AUTOGLASS/PDF/{today}',exist_ok=True)
        for pdf in pdffnlst:
            basename=os.path.basename(pdf)
            newbase=f'{folder}_{basename}'
            newpdf=f'/data/transient/AUTOGLASS/PDF/{today}/{newbase}'
            if not os.path.exists(newpdf):
                os.system(f'cp {pdf} {newpdf}')
    
def run_pipeline(rawdir,bands,today,template,opstep_path,pixel_scale,
                 solve_field_cmd,sextractor_cmd,scamp_cmd,swarp_cmd,hotpants_cmd,
                 run_cpu,xregion,yregion,telescope_name,folder=None,cppdf=False):

    start_all = time.perf_counter()
    tag('BEGIN')
    today=today.strftime('%Y-%m-%d')
    print(f'THE FOLDERS NEED TO BE PROCESSED : \n{today}')
    raw_date_folder=rawdir+'/'+today
    print(f'THE RAW DATA LOCATION  : {raw_date_folder}')

    all_rawdata=glob.glob(f'{raw_date_folder}/*/*')
    
    if telescope_name=='xinglongf203' and len(all_rawdata)>0:
        for rawdata in all_rawdata:
            overscan_cut(rawdata)
            
    sizes=Counter([os.stat(all_rawdata[i]).st_size for i in range(len(all_rawdata))])
    all_rawdata=glob.glob(f'{raw_date_folder}/*/*')
    if len(all_rawdata)>0:
        sizes=Counter([os.path.getsize(all_rawdata[i]) for i in range(len(all_rawdata))])
        normal_size=sizes.most_common(1)[0][0]
        for i in range(len(all_rawdata)):
            size=os.path.getsize(all_rawdata[i])
            if size!=normal_size:
                check_upload(all_rawdata[i])
            if size!=normal_size:
                os.remove(all_rawdata[i])

    today_time=pd.to_datetime(extract_date(raw_date_folder))
    all_date_folders=glob.glob(f'{rawdir}/*')
    all_date=[extract_date(date_folder) for date_folder in all_date_folders]
    all_date=pd.to_datetime(sorted(all_date))
    pass_day_true=today_time>=all_date
    
    pass_day=all_date[np.where(pass_day_true)[0]]
    
    global closest_day_bias
    global closest_day_flat
    for i in range(len(pass_day)-1,-1,-1):
        pass_dayi=pass_day[i].strftime('%Y-%m-%d')
        pass_day_folder=rawdir+'/'+pass_dayi
        pass_day_bias=glob.glob(f'{pass_day_folder}/BIAS/*')
        if len(pass_day_bias)!=0:
            closest_day_bias=pass_day_folder
            break

    for i in range(len(pass_day)-1,-1,-1):
        pass_dayi=pass_day[i].strftime('%Y-%m-%d')
        pass_day_folder=rawdir+'/'+pass_dayi
        pass_day_flat=glob.glob(f'{pass_day_folder}/FLAT/*')
        if len(pass_day_flat)!=0:
            closest_day_flat=pass_day_folder
            break

    lightfnlst=glob.glob(f'{raw_date_folder}/LIGHT/*')
    if len(lightfnlst)==0:
        print(f'THERE IS NO LIGHT IN {today}, SKIP!!!')

    elif len(lightfnlst)!=0:
        biasfnlst=glob.glob(f'{raw_date_folder}/BIAS/*')
        if len(biasfnlst)==0:
            print(f'THERE IS NO BIAS IN {raw_date_folder}, COPY BIAS FROM {closest_day_bias}')
            os.system(f'cp -r {closest_day_bias}/BIAS {raw_date_folder}/')

        flatfnlst=glob.glob(f'{raw_date_folder}/FLAT/*')
        if len(flatfnlst)==0:
            print(f'THERE IS NO FLAT IN {raw_date_folder}, COPY FLAT FROM {closest_day_flat}')
            os.system(f'cp -r {closest_day_flat}/FLAT {raw_date_folder}/')

        print('LIGHT EXISTS, BEGIN PREPROCESSING ...')
        start_all1 = time.perf_counter()
        
        all_rawdata=glob.glob(f'{raw_date_folder}/*/*')
        
        if telescope_name=='xinglongf203' and len(all_rawdata)>0:
            for rawdata in all_rawdata:
                overscan_cut(rawdata)
        
        preprocessing(raw_date_folder,bands,opstep_path,pixel_scale,
                      solve_field_cmd,sextractor_cmd,scamp_cmd,swarp_cmd,
                      run_cpu,xregion,yregion)
        end_all1 = time.perf_counter()
        runTime_all1 = end_all1 - start_all1
        print(f"{raw_date_folder} ALL PREPROCESSING SPENT TIME : {round(runTime_all1,3)} seconds"+
            "\nINCLUDING : BIASCOMBINE, FLATCOMBINE, PREPROCESS, PHOTOMETRY, ASTROMETRY, IMAGECOMBINE")

    start_all2 = time.perf_counter()
    imagesubtraction(raw_date_folder,template,opstep_path,pixel_scale,
                     swarp_cmd,hotpants_cmd,run_cpu,xregion,yregion,folder,cppdf,today)
    end_all2 = time.perf_counter()
    runTime_all2 = end_all2 - start_all2
    print(f"{raw_date_folder} ALL IMAGESUBTRATION SPENT TIME : {round(runTime_all2,3)} seconds"+
        "\nINCLUDING : WCSCATALOG, REPROJECT, IMAGESUBTRATE, PLOTPDF")

    end_all = time.perf_counter()
    runTime_all = end_all - start_all
    if len(lightfnlst)!=0:
        mean_runTime=runTime_all/len(lightfnlst)
    else:
        mean_runTime=0

    os.system(f'rm {opstep_path}/coadd.weight.fits')
    os.system(f'rm {opstep_path}/*.jpg')
    
    tag('END')
    print('{0:-^96}'.format(''))
    print('{0:-^96}'.format('WE HAVE DONE ALL PROGRAM WITH THE DATA IN'))
    print('{0:-^96}'.format(f'{raw_date_folder}'))
    print('{0:!^96}'.format(''))
    print('{0:-^96}'.format(f'ALL PROGRAM SPENT TIME : {round(runTime_all,3)} SEC'))
    print('{0:-^96}'.format(f'THERE IS {len(lightfnlst)} OBSERVED FRAMES'))
    print('{0:-^96}'.format(f'MEAN TIME : {round(mean_runTime,3)} SEC / FRAME'))
    print('{0:-^96}'.format('SEE YOU TOMORROW !!!'))
    print('{0:-^96}'.format(f'END TIME (UTC) : {datetime.now()}'))
    