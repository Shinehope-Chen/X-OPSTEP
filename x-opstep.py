import argparse

description='This is a pipeline for xinglong nearby galaxies transient survey'
base_path=None
process_mode=None
process_date_begin=None
process_date_end=None
run_cpu=None
telescope_names=None

parser= argparse.ArgumentParser(description=description)
parser.add_argument('--base_path', '-bp', required=False,type=str,
	help='the path you save data')
parser.add_argument('--process_mode', '-pm', required=False,type=str,
	help='choose read_date or set_date, read_date will read the system date, set_date can set the date by yourself')
parser.add_argument('--process_date_begin', '-pdb', required=False,type=str,
	help='the day you want to start processing, only work when process_mode is set_date, format YYYY-MM-DD')
parser.add_argument('--process_date_end', '-pde', required=False,type=str,
	help='the day you want to end processing, only work when process_mode is set_date, format YYYY-MM-DD')
parser.add_argument('--run_cpu', '-ncpu', required=False,type=str,
	help='how many cpu you want to use')
parser.add_argument('--telescope_names', '-tn', required=False,type=str,
	help='which telescope data you want to process')
args=parser.parse_args()

process_mode=args.process_mode
process_date_begin=args.process_date_begin
process_date_end=args.process_date_end
run_cpu=args.run_cpu
telescope_names=args.telescope_names
base_path=args.base_path

import xopstepbase as xob
from astropy import units as u
import sys
from datetime import datetime, timedelta
import pandas as pd
import os
from astropy.io import fits
import glob
import warnings
warnings.filterwarnings('ignore')

class Logger(object):
    def __init__(self, filename='pipeline.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

opstep_path = os.path.dirname(os.path.abspath(__file__))
config=open('/data/transient/AUTOGLASS/x-opstep.config')
config_data=config.readlines()
if not base_path:
    base_path=config_data[0].split()[1]
bands=config_data[1].split()[1].split(',')
obs_loc=config_data[2].split()[1].split(',')
obs_location=[float(obs_loc[0])*u.deg, float(obs_loc[1])*u.deg,float(obs_loc[2])*u.m]
if not process_mode:
    process_mode=config_data[3].split()[1]
if not process_date_begin:
    process_date_begin=config_data[4].split()[1]
if not process_date_end:
    process_date_end=config_data[5].split()[1]
solve_field_cmd=config_data[6].split()[1]
sextractor_cmd=config_data[7].split()[1]
scamp_cmd=config_data[8].split()[1]
swarp_cmd=config_data[9].split()[1]
hotpants_cmd=config_data[10].split()[1]
if not run_cpu:
    run_cpu=int(config_data[11].split()[1])
else:
    run_cpu=int(args.run_cpu)
xregion=int(config_data[12].split()[1])
yregion=int(config_data[13].split()[1])

telescope_name_list={}
for data in config_data:
    if 'TELESCOPE_NAME' in data:
        key=data.split()[3].split('Survey_')[-1]
        telescope_name_list[key]=data

if not telescope_names or telescope_names=='all':
    telescope_name_toprocess=telescope_name_list.copy()
    # print(telescope_name_toprocess)
else:
    telescope_name_toprocess={}
    for tn in telescope_names.split(','):
        if tn in telescope_name_list.keys():
            telescope_name_toprocess[tn]=telescope_name_list[tn]
        else:
            print(f'{tn} is not in the config file')
    # print(telescope_name_toprocess)
    
if base_path!='/data/transient/AUTOGLASS':

    # 读取符合日期格式的文件夹
    rawdir=f'{base_path}/rawdir'
    reddir=f'{base_path}/reddir'
    template=f'{base_path}/template'
    os.makedirs(reddir, exist_ok=True)

    telescope_name=fits.getheader(glob.glob(f'{rawdir}/*/LIGHT/*')[0])['TELESCOP']

    for data in config_data:
        if 'TELESCOPE_NAME' in data:
            if data.split()[1]==telescope_name:
                pixel_scale=float(data.split('pixel_scale')[-1])

    # 设置用户输入的日期
    if process_mode=='set_date':
        y1,m1,d1=process_date_begin.split('-')
        y2,m2,d2=process_date_end.split('-')
        start_date = datetime(int(y1),int(m1),int(d1))
        end_date = datetime(int(y2),int(m2),int(d2))
        delta = timedelta(days=1)

    if process_mode=='read_date':
        start_date = datetime.now()
        end_date = datetime.now()
        delta = timedelta(days=1)

    for date in range(int((end_date - start_date).days) + 1):
        current_date = start_date + delta * date

        if not os.path.exists(f"{reddir}/{current_date.strftime('%Y-%m-%d')}"):
            os.system(f"mkdir {reddir}/{current_date.strftime('%Y-%m-%d')}")

        # 输出日志
        sys.stdout = Logger(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/pipeline_out_{current_date.strftime('%Y-%m-%d')}.log", sys.stdout)
        sys.stderr = Logger(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/pipeline_err_{current_date.strftime('%Y-%m-%d')}.log", sys.stderr)

        xob.run_pipeline(rawdir,bands,current_date,template,opstep_path,pixel_scale,
                                solve_field_cmd,sextractor_cmd,scamp_cmd,swarp_cmd,hotpants_cmd,
                                run_cpu,xregion,yregion,telescope_name,folder=None,cppdf=False)

        if not os.path.exists(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/bias.fits"):
            os.system(f"rm -r {reddir}/{current_date.strftime('%Y-%m-%d')}")

if base_path=='/data/transient/AUTOGLASS':
    for key in telescope_name_toprocess:
        value=telescope_name_toprocess[key]
        folder=value.split()[3]
        telescope_name=value.split()[1]
        pixel_scale=float(value.split()[-1])
#     for data in config_data:
#         if 'TELESCOPE_NAME' in data and data[0]!='#':
#             folder=data.split()[3]
            # telescope_name=data.split()[1]
#             pixel_scale=float(data.split()[-1])

        # 读取符合日期格式的文件夹
        rawdir=f'{base_path}/{folder}/rawdir'
        reddir=f'{base_path}/{folder}/reddir'
        template=f'{base_path}/{folder}/template'
        os.makedirs(reddir, exist_ok=True)
        print(folder,telescope_name,pixel_scale)
                    
        # 设置用户输入的日期
        if process_mode=='set_date':
            y1,m1,d1=process_date_begin.split('-')
            y2,m2,d2=process_date_end.split('-')
            start_date = datetime(int(y1),int(m1),int(d1))
            end_date = datetime(int(y2),int(m2),int(d2))
            delta = timedelta(days=1)

        if process_mode=='read_date':
            start_date = datetime.now()
            end_date = datetime.now()
            delta = timedelta(days=1)

        for date in range(int((end_date - start_date).days) + 1):
            current_date = start_date + delta * date

            if not os.path.exists(f"{reddir}/{current_date.strftime('%Y-%m-%d')}"):
                os.system(f"mkdir {reddir}/{current_date.strftime('%Y-%m-%d')}")
            
            # 输出日志
            sys.stdout = Logger(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/pipeline_out_{current_date.strftime('%Y-%m-%d')}.log", sys.stdout)
            sys.stderr = Logger(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/pipeline_err_{current_date.strftime('%Y-%m-%d')}.log", sys.stderr)

            xob.run_pipeline(rawdir,bands,current_date,template,opstep_path,pixel_scale,
                            solve_field_cmd,sextractor_cmd,scamp_cmd,swarp_cmd,hotpants_cmd,
                            run_cpu,xregion,yregion,telescope_name,folder=folder,cppdf=True)
            
            if not os.path.exists(f"{reddir}/{current_date.strftime('%Y-%m-%d')}/bias.fits"):
                os.system(f"rm -r {reddir}/{current_date.strftime('%Y-%m-%d')}")