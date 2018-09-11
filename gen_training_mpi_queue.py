import numpy as np
import csv
import time
import requests
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
from shp_utils import *
import os
from clip import *
#import multiprocessing as mp
os.environ["PATH"] += os.pathsep + "/soge-home/staff/hert1424/.conda_envs/parallel/bin/"

tic = time.time()

def get_granule(pt, download=False):
  #which granule? Most recent with cloud <30%


  api = SentinelAPI('lucas.kruitwagen', 'Passw0rd', 'https://scihub.copernicus.eu/dhus')

  # download single scene by known product id
  #api.download('d4b5d3ca-8d93-4f91-b0d5-3ab194f08d80')

  #https://scihub.copernicus.eu/dhus/odata/v1/Products('d4b5d3ca-8d93-4f91-b0d5-3ab194f08d80')/$value

  #geojson = 'aoi.geojson'
  #print geojson
  #gj = read_geojson(geojson)
  #print gj
  #print geojson_to_wkt(read_geojson(geojson))
  # search by polygon, time, and Hub query keywords
  #footprint = geojson_to_wkt(read_geojson(geojson))

  footprint = 'POINT('+str(pt[1])+' '+str(pt[0])+')'


  #str_3 = '(beginPosition:[2015-12-29T00:00:00Z TO NOW]) AND (cloudcoverpercentage:[0 TO 30]) AND (endPosition:[2017-01-01T00:00:00Z TO NOW]) AND (platformname:Sentinel-2) AND (footprint:"Intersects(POLYGON((-1.260888700361533 51.75293095659052,-1.2562874323233975 51.75293095659052,-1.2562874323233975 51.755951920165415,-1.260888700361533 51.755951920165415,-1.260888700361533 51.75293095659052)))")'
  #print str_3
  #products = api.query(raw = str_q)
  #print products
  products = api.query(footprint,
                     date = ('20161109', '20171109'),
                     producttype = 'S2MSI1C',
                     platformname = 'Sentinel-2',
                     cloudcoverpercentage = (0, 30.0))

  product_ids = products.keys()
  prod_list = [products[el] for el in product_ids]
  #print len(prod_list)
  #print [el['uuid'] for el in prod_list]
  prod_list = sorted(prod_list, key=lambda k: k['cloudcoverpercentage']) 
  #for el in prod_list:
  # 	print el['uuid'], el['beginposition'], el['cloudcoverpercentage']
  attribs = products[product_ids[0]].keys()
  img_id = prod_list[0]['uuid']
  #print products[img_id]['filename']
  json_out = api.to_geojson(products)
  #print json_out
  if download:
    result = None
    while result<0:
      try:
        result = api.download(prod_list[0]['uuid'])
        print 'result',result,result<0
      except:
        time.sleep(60)
  return products[img_id]['filename']

def gen_training(fname,pt,ID,q=None):
  print 'hi', ID
  #q.put([ID,'helllllooo','how about this fname'])
  #time.sleep(5)
  #return 1
  
  try:
    rwd = os.getcwd()
    #os.chdir(os.path.join(rwd,'GRANULES')) #should be DPHIL/GRANULES/
    print 'rwd',rwd
  
    #make a new dir per dl
    try:
      os.mkdir(os.path.join(rwd,'GRANULES',ID))
      print "mkdir :"+os.path.join(rwd,'GRANULES',ID)
    except:
      print 'dir already exosts'
    #os.chdir(os.path.join(rwd,'GRANULES',ID))
    #get the granule
    #fname = get_granule(pt, download=False)
    print 'filename:', fname
  
    fname_split = fname.split('_')
    #print fname, 'L1C_'+'_'.join(fname_split[-2:])[:-5]
  
    #unzip the granule
    'unzipping...'
    try:
      command = ['unzip',os.path.join(rwd,glob.glob('*'+fname[:-5]+'*')[0]), '-d', os.path.join(rwd,'GRANULES',ID)]
      #print ' '.join(command)
      subprocess.check_call(' '.join(command), shell=True)
    except:
      print 'no unzippy'
  
    #ranme the file
    #command = ['mv', os.path.join(rwd,'GRANULES',fname), os.path.join(rwd,'GRANULES',ID+'_'+fname)]
    #subprocess.call(' '.join(command), shell=True)

    #remove the zip
    try:
      print 'removing zip...'
      os.remove(fname[:-5]+'.zip')
    except:
      print 'nothing to remove?'
    #rename fname

    #fname = ID+'_'+fname 
    #chaange to Granules folder
    os.chdir(os.path.join(rwd,'GRANULES',ID))
    print os.getcwd()
    #exit()  
    #get gran_dir name
    os.chdir(os.path.join(rwd,'GRANULES',ID,fname,'GRANULE'))
    gran_dir = glob.glob('*')[0]
    #print 'gran_dir', gran_dir
    os.chdir(os.path.join(rwd,'GRANULES',ID))
    
    #gen extra samples
    print 'generating samples...'
    samples = gen_samples(pt,fname,rwd,ID,gran_dir,4)
    print 'samples:',samples
  
    #clip L1C
    print 'clipping L1C to samples...'
    clipL1_dir = os.path.join(rwd,'GRANULES',ID,fname,'GRANULE',gran_dir,'IMG_DATA')
    clip_orig(clipL1_dir,pt,samples)
  
    #upgrade it to L2A
    os.chdir(os.path.join(rwd,'GRANULES',ID))
    #print 'upgrading to L2A using sen2cor...'
    upgrade_L2A(fname, rwd,ID,gran_dir)
    
    
    #clip L2A
    if 'OPER' in fname:
      atm_corr_fname = fname.replace('OPER','USER')
      atm_corr_fname = atm_corr_fname.replace('L1C','L2A')
      gran_dir_atm_corr = gran_dir.replace('OPER','USER')
      gran_dir_atm_corr = gran_dir_atm_corr.replace('L1C','L2A')
    else:
      atm_corr_fname = fname.replace('L1C','L2A')
      gran_dir_atm_corr = gran_dir.replace('L1C','L2A')
    
    print 'clipping L2A to samples ...'
    clipL2_dir = os.path.join(rwd,'GRANULES',ID,atm_corr_fname,'GRANULE',gran_dir_atm_corr,'IMG_DATA')
    clip_atm_corr(clipL2_dir,pt,samples)
  
    #wipe all originals
    print 'wiping originals to save disk'
    wipe(clipL1_dir,clipL2_dir)
    
    #reset DIR
    os.chdir(rwd)   
    
    print 'exit dir:', os.getcwd()
    res = [ID,1,fname]
    q.put(res)
    return 1
  except Exception as e:
    os.chdir(rwd)
    print e
    res = [ID,e,fname]
    print(res)
    return 0
  

def upgrade_L2A(fname,rwd,ID,gran_dir):
  #rwd = os.getcwd()
  #print 'rwd_L2a', rwd
  os.chdir(os.path.join(rwd,'GRANULES',ID,fname, 'GRANULE'))
  print 'cwd for L2A', os.getcwd()
  target = os.path.join(rwd,'GRANULES',ID,fname,'GRANULE',gran_dir)
  print 'target',target
  command = ['L2A_Process', target,'--resolution=60']
  #subprocess.call(' '.join(command), shell=True)
  subprocess.check_call(' '.join(command), shell=True)
  #print c
  #if c!=0:
  #  raise Exception(stderr)
  command = ['L2A_Process', target,'--resolution=10']
  print command
  subprocess.check_call(' '.join(command), shell=True)
  #stdout, stderr = c.communicate()
  #print stdout, stderr
  #if c.returncode!=0:
  #  raise Exception(stderr)
  os.chdir(rwd)

def gen_samples(pt,fname,rwd,ID,gran_dir,add_samples):

  fname_split = fname.split('_')
  os.chdir(os.path.join(rwd,'GRANULES',ID,fname,'GRANULE',gran_dir,'IMG_DATA'))
  files = glob.glob('*.jp2')




  src = gdal.Open(files[0])
  ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
  lrx = ulx + (src.RasterXSize * xres)
  lry = uly + (src.RasterYSize * yres)
  print ulx, uly, lry, lrx
  Dx = lrx - ulx
  Dy = uly - lry

  source = osr.SpatialReference()
  source.ImportFromWkt(src.GetProjection())

  # The target projection
  target = osr.SpatialReference()
  target.ImportFromEPSG(4326)
  # Create the transform - this can be used repeatedly
  transform = osr.CoordinateTransformation(source, target)
  # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
  #lon/lat
  UL = transform.TransformPoint(ulx, uly)
  BR = transform.TransformPoint(lrx, lry)
  max_lat = UL[1]
  min_lat = BR[1]
  max_lon = BR[0]
  min_lon = UL[0]

  #print 'max_lat',max_lat,'min_lat',min_lat,'max_lon',max_lon,'min_lon',min_lon
  count=0

  samples = [pt]
  while len(samples)<(add_samples+1):
    ll = (int(np.random.rand()*Dx + ulx),int(np.random.rand()*Dy + lry))
    ll = transform.TransformPoint(ll[0],ll[1])
    lon = ll[0]
    lat = ll[1]

    flag = False
    
    #test edges
    dt = V_inv((lat,lon),(lat,max_lon))[0]
    db = V_inv((lat,lon),(lat,min_lon))[0]
    dl = V_inv((lat,lon),(min_lat,lon))[0]
    dr = V_inv((lat,lon),(max_lat,lon))[0]
    #print dt,db,dl,dr

    flag = dt<5.0 or db<5.0 or dl<5.0 or dr<5.0
    #test points
    for sample in samples:
      #they're in lat/lon
      dx = V_inv((lat,lon),(lat,sample[1]))[0]
      dy = V_inv((lat,lon),(sample[0],lon))[0]
      #print sample, dx, dy
      flag+= dx<5.0 or dy<5.0
      #if flag ==True:
        #print dx,dy
    #print 'flag', flag

    if flag==True:
      count+=1
      #print count
    elif flag==False:
      samples.append((ll[1],ll[0]))
  #print count

  os.chdir(os.path.join(rwd,'GRANULES','META'))
  metaname = ID+'.txt'
  #print metaname
  with open(metaname, "wb") as myfile:
    count=0
    for sample in samples:
      myfile.write(str(count)+','+str(sample[0])+','+str(sample[1])+','+fname+'\n')
      count+=1
  os.chdir(os.path.join(rwd))

  return samples

def wipe(clipL1_dir,clipL2_dir):
  path_10 = os.path.join(clipL2_dir, 'R10m')
  path_20 = os.path.join(clipL2_dir, 'R20m')
  path_60 = os.path.join(clipL2_dir, 'R60m')
  path_orig = os.path.join(clipL1_dir)
  wipe_dir(path_10)
  wipe_dir(path_20)
  wipe_dir(path_60)
  wipe_dir(path_orig)

def wipe_dir(dirr):
  print 'wiping...', dirr
  #rwd = os.getcwd()
  os.chdir(dirr)
  files = glob.glob('*.jp2')
  for f in files:
    os.unlink(f)
  #os.chdir(rwd)

def csv_dict_out(listodicts,fname,headers):
  print 'writing csv...'
  f = open(fname,'wb')
  w = csv.DictWriter(f,headers)
  w.writeheader()
  w.writerows(listodicts)
  f.close()

def read_csv_dict(filename):
  out=[]
  dic={}
  with open(filename,'rb') as infile:
    reader = csv.reader(infile)
    headers = next(reader)
    #print headers
    for row in reader:
      dic = {key: value for key, value in zip(headers, row)}
      out.append(dic)
  #print 'years'
  infile.close()

  return out,headers

def listener(q):
  """ listens for messages on q, write to file"""

  cont,headers = read_csv_dict('progress.csv')
  count=0
  while 1:
    m = q.get()
    #print m
    if m[1]=='kill':
      print m, time.time()
      break
    elif m:
      print 'incoming message!', m
      print time.time()
      ID = m[0]
      print ID
      message = m[1]
      fname = m[2]
      (el for el in cont if el['Tracker_ID']==ID).next()['message']=message
      print message
      (el for el in cont if el['Tracker_ID']==ID).next()['filename']=fname
      csv_dict_out(cont,'progress.csv',headers)
      count+=1
      print count

def test():
  test_pt = (49.8232,18.47851389) 
  #test_pt = (-51.5460148, -72.2312558)
  print gen_training(test_pt,'G105604')

def director():

  #if __name__ == "__main__":
  #fetch first point

  
  cont,headers = read_csv_dict('progress.csv')
  pts = [[(float(el['Latitude']),float(el['Longitude'])),el['Tracker_ID']] for el in cont if el['message']=='']

  #pts = pts[0:12]
  print pts
  print len(pts)
  manager = mp.Manager()
  q = manager.Queue()
  pool = mp.Pool()

  watcher = pool.apply_async(listener,args=(q,))
  
  jobs = []
  for pt in pts:
    job = pool.apply_async(gen_training,args=(pt[0],pt[1],q))
    jobs.append(job)

  for job in jobs:
    job.get()

  q.put(['null','kill','null'])
  #q.join()
  #pool.join()
  pool.close()
  pool.join()
  #q.put(['null','kill','null'])
  #pool.join()
  #get granule
  #make some clips - how many? 4 mehb? 
  #add to meta
  #upgrade to lvl 2A via sen2cor
  #make same clips
  #add to meta
  #delete
  
#if __name__ == "__main__":
  #director()

from mpi4py import MPI

comm = MPI.COMM_WORLD

print "Reading progress.csv"
cont,headers = read_csv_dict('progress.csv')
pts = [[(float(el['Latitude']),float(el['Longitude'])),el['Tracker_ID']] for el in cont if el['message']=='']

myrank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
procnm = MPI.Get_processor_name()

if (myrank==0):
  print pts
  print len(pts)
  rank=1
  for pt in pts:
    if (rank==nprocs):
       rank=1
    fname = get_granule(pt[0], download=True)
    dat=[fname,pt[0],pt[1]]
    comm.send(dat, dest=rank, tag=11)
else:
    while True:
      dat=comm.recv(source=0, tag=11)
      gen_training(dat)



comm.Barrier()

