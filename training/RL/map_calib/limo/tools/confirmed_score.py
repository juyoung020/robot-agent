import json,sys,numpy as np
run,gtp=sys.argv[1],sys.argv[2]
fr=json.load(open(run+'/frame.json')) if len(sys.argv)<4 else None
c,s,tx,ty=(fr['map_from_world'] if fr else json.load(open(sys.argv[3]))['map_from_world'])
def w2m(p): return np.array([c*p[0]-s*p[1]+tx, s*p[0]+c*p[1]+ty, p[2]])
gt=[o for o in json.load(open(gtp)) if o['category'] not in ('walls','floors','ceilings','lawn','paver','driveway','roof','agent')]
# GT AABB in map frame: rotate 8 corners and take AABB
def m2w(p):
    x,y=p[0]-tx,p[1]-ty; return np.array([c*x+s*y, -s*x+c*y, p[2]])
for o in gt:
    o['mlo'],o['mhi']=np.array(o['lo']),np.array(o['hi']); o['mc']=0.5*(o['mlo']+o['mhi'])
STRICT={'table':{'breakfast_table','coffee_table'},'lamp':{'downlight','room_light','track_light','garden_light'},'shelf':{'shelf'},
 'sofa':{'sofa'},'radio receiver':{'radio'},'picture frame':{'picture'},'plant':{'bush','tree'},'door':{'door'},'window':{'fixed_window'},
 'light switch':{'electric_switch'},'electric outlet':{'wall_socket'},'staircase':{'stairs'},'railing':{'rail_fence'},
 'chair':{'straight_chair','garden_chair'},'couch':{'sofa'},'dining table':{'breakfast_table','coffee_table'},'potted plant':{'bush','tree'},
 'tv':{'wall_mounted_tv'},'microwave':{'microwave'},'oven':{'oven'},'refrigerator':{'fridge'},'sink':{'drop_in_sink'},'car':{'car'}}
LEN={'table':{'bar','countertop'},'shelf':{'top_cabinet','bottom_cabinet','hall_tree'},'dining table':{'bar','countertop'},'chair':{'bar'},'lamp':{'standing_mirror'} and set()}
def boxdist(p,lo,hi): return np.linalg.norm(np.maximum(0,np.maximum(lo-p,p-hi)))
d=json.load(open(run+'/memory/scene.json'))
res=[];
for n in d['nodes']:
    a=n['attributes']
    if n.get('layer')!=2 or 'bounding_box' not in a or 'metadata' not in a or a['metadata'].get('n_obs') is None: continue
    if a['metadata'].get('state')=='gone': continue
    bcm=np.array(a['bounding_box']['world_P_center']); bdm=np.array(a['bounding_box']['dimensions'])
    cs=np.array([m2w(bcm+np.array([sx,sy,sz])*bdm/2) for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)])
    bc=m2w(bcm); blo,bhi=cs.min(0),cs.max(0); bd=bhi-blo
    near=[]
    for o in gt:
        ov=np.prod(np.clip(np.minimum(bhi,o['mhi'])-np.maximum(blo,o['mlo']),0,None))/max(np.prod(np.maximum(bd,0.02)),1e-9)
        if ov>=0.2 or boxdist(bc,o['mlo'],o['mhi'])<=0.10: near.append((boxdist(bc,o['mlo'],o['mhi']),o))
    lab=a['name']
    st=[o for _,o in near if o['category'] in STRICT.get(lab,set())]
    le=[o for _,o in near if o['category'] in LEN.get(lab,set())]
    if st: k='strict'; m=st
    elif le: k='lenient'; m=le
    elif near: k='mislabel'; m=[]
    else: k='false'; m=[]
    if m:
        o=min(m,key=lambda o:np.linalg.norm(o['mc']-bc)); err=np.linalg.norm(o['mc']-bc); exy=np.linalg.norm((o['mc']-bc)[:2]); gn=o['name']; gext=o['mhi']-o['mlo']
    else: err=exy=np.nan; gn=','.join(sorted({o['category'] for _,o in near}))[:60]; gext=None
    res.append((k,lab,err,exy))
    print('%-9s %-15s n_obs %3d  c %s  err3d %.2f xy %.2f  gt %s %s'%(k,lab,a['metadata']['n_obs'],np.round(bc,2),err,exy,gn, '' if gext is None else np.round(gext,2)))
import collections; print(collections.Counter(r[0] for r in res), 'n', len(res))
e=np.array([r[2] for r in res if r[0] in('strict','lenient')]); exy=np.array([r[3] for r in res if r[0] in('strict','lenient')])
if len(e): print('pos err (correct, n=%d): 3d median %.3f mean %.3f rms %.3f max %.3f | xy median %.3f rms %.3f'%(len(e),np.median(e),e.mean(),np.sqrt((e**2).mean()),e.max(),np.median(exy),np.sqrt((exy**2).mean())))
