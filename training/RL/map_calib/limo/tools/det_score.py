import sys,json,collections,numpy as np, sgrec as rec
recp, gtp = sys.argv[1], sys.argv[2]
P,G,I = rec.load(recp)
coco=[l.strip() for l in open('/home/juyoung/behavior-2026-limo/src/scene_graph/ovdet/config/coco80.txt')]
EXC={'walls','floors','ceilings','lawn','paver','driveway','roof','agent'}
gt=[o for o in json.load(open(gtp)) if o['category'] not in EXC]
for o in gt: o['lo']=np.array(o['lo']); o['hi']=np.array(o['hi']); o['c']=(o['lo']+o['hi'])/2
STRICT={'chair':{'straight_chair','garden_chair'},'couch':{'sofa'},'dining table':{'breakfast_table','coffee_table'},
 'tv':{'wall_mounted_tv'},'microwave':{'microwave'},'oven':{'oven'},'refrigerator':{'fridge'},'sink':{'drop_in_sink'},'car':{'car'},
 'potted plant':set(),'bench':{'garden_chair'} and set(),'clock':set(),'vase':set(),'book':set()}
LEN={'chair':{'bar'},'dining table':{'bar','countertop'},'potted plant':{'bush','tree'},'bench':set()}
VOCAB_GT={'straight_chair','garden_chair','sofa','breakfast_table','coffee_table','wall_mounted_tv','microwave','oven','fridge','drop_in_sink','car'}
Gs=np.array(G)
def pose_at(t):
    i=np.searchsorted(Gs[:,0],t); i=min(max(i,1),len(Gs)-1); a,b=Gs[i-1],Gs[i]
    u=0 if b[0]==a[0] else np.clip((t-a[0])/(b[0]-a[0]),0,1)
    dy=(b[3]-a[3]+np.pi)%(2*np.pi)-np.pi
    return a[1]+u*(b[1]-a[1]), a[2]+u*(b[2]-a[2]), a[3]+u*dy
OFF=np.array([0.094,0.0,0.18053])  # cam in base footprint frame (x fwd, z world height); from gt_poses.csv
def cam(t):
    x,y,yaw=pose_at(t); c,s=np.cos(yaw),np.sin(yaw)
    Rb=np.array([[c,-s,0],[s,c,0],[0,0,1]]); p=np.array([x,y,0])+Rb@np.array([OFF[0],OFF[1],0]); p[2]=OFF[2]
    # optical frame: x right, y down, z fwd  -> body: fwd=z, left=-x, up=-y
    Ro=Rb@np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    return p,Ro
def boxdist(p,lo,hi): return np.linalg.norm(np.maximum(0,np.maximum(lo-p,p-hi)),axis=-1)
stats=collections.Counter(); perkf=[]; floorz=[]; miss=collections.defaultdict(lambda:[0,0]); cmiss=collections.defaultdict(lambda:[0,0])
lat=collections.defaultdict(list); fpname=collections.Counter(); misname=collections.Counter()
lag_st=None; VIS=[]
for k,d in enumerate(I):
    st=d['st']; p,Ro=cam(st); fx,fy,cx,cy=d['K']; D=d['depth']; h,w=D.shape
    # floor check: bottom rows
    vv,uu=np.mgrid[410:520:6, 0:w:12]; z=D[vv,uu]; ok=(z>0.3)&(z<1.2)
    pts=np.stack([(uu[ok]-cx)/fx*z[ok],(vv[ok]-cy)/fy*z[ok],z[ok]],1)@Ro.T+p; floorz+=list(pts[:,2])
    nfp=0; hit=set()
    for j in range(d['n']):
        m=np.unpackbits(d['bits'][j].view(np.uint8),bitorder='little')[:d['mw']*d['mh']].reshape(d['mh'],d['mw'])
        ys,xs=np.nonzero(m); sx,sy,ox,oy=d['ms']
        u=np.clip((xs*sx+ox+sx/2).astype(int),0,w-1); v=np.clip((ys*sy+oy+sy/2).astype(int),0,h-1)
        z=D[v,u]; good=(z>0.05)&(z<8)&np.isfinite(z)
        if good.sum()<10: stats['nodepth']+=1; continue
        z=z[good]; u=u[good]; v=v[good]
        med=np.median(z); mad=np.median(np.abs(z-med))*1.4826+0.02; keep=np.abs(z-med)<3*mad
        X=np.stack([(u[keep]-cx)/fx*z[keep],(v[keep]-cy)/fy*z[keep],z[keep]],1)@Ro.T+p
        name=coco[d['cls'][j]]
        best=None;bf=0
        for gi,o in enumerate(gt):
            f=(np.all(X>=o['lo']-0.05,1)&np.all(X<=o['hi']+0.05,1)).mean()
            if f>bf: bf,best=f,gi
        cands=[gi for gi,o in enumerate(gt) if (np.all(X>=o['lo']-0.05,1)&np.all(X<=o['hi']+0.05,1)).mean()>=0.3]
        dist=np.linalg.norm(np.median(X,0)-p)
        if not cands: k2='false'; nfp+=1; fpname[name]+=1
        else:
            cats={gt[gi]['category'] for gi in cands}
            if cats & STRICT.get(name,set()): k2='strict'
            elif cats & LEN.get(name,set()): k2='lenient'
            else: k2='mislabel'; misname[(name,tuple(sorted(cats)))]+=1
            for gi in cands: hit.add(gi)
            if k2 in('strict','lenient'):
                gi=[g for g in cands if gt[g]['category'] in STRICT.get(name,set())|LEN.get(name,set())][0]; lat[gi].append(np.median(X,0))
        stats[k2]+=1
    perkf.append(nfp)
    # visibility for misses
    for gi,o in enumerate(gt):
        if o['category'] not in VOCAB_GT: continue
        pc=Ro.T@(o['c']-p)
        dist=np.linalg.norm(o['c']-p)
        if pc[2]<0.3 or dist>6: continue
        g=np.linspace(0.1,0.9,4); S=np.array([[a,b,c] for a in g for b in g for c in g]); S=o['lo']+S*(o['hi']-o['lo'])
        Q=(S-p)@Ro; zq=Q[:,2]; inf=zq>0.1
        uq=np.where(inf,fx*Q[:,0]/np.maximum(zq,1e-3)+cx,-1); vq=np.where(inf,fy*Q[:,1]/np.maximum(zq,1e-3)+cy,-1)
        ins=inf&(uq>=0)&(uq<w)&(vq>=0)&(vq<h)
        if ins.sum()==0: continue
        bw=uq[ins].max()-uq[ins].min(); bh=vq[ins].max()-vq[ins].min()
        if bw<20 or bh<20 or bw*bh<1600: continue
        vis=ins.copy(); vis[ins]=D[vq[ins].astype(int),uq[ins].astype(int)]>=zq[ins]-0.10
        if vis.sum()<0.5*len(S): continue
        b=0 if dist<1.5 else (1 if dist<2.5 else (2 if dist<3.0 else 3))
        m=gi not in hit
        VIS.append((k,o['name'],round(dist,2),m))
        miss[b][0]+=m; miss[b][1]+=1; cmiss[o['category']][0]+=m; cmiss[o['category']][1]+=1
print('keyframes',len(I),'dets',sum(d['n'] for d in I),dict(stats))
fz=np.array(floorz); print('floor z median %.3f (n %d)'%(np.median(fz),len(fz)))
pk=np.array(perkf); print('FP per keyframe mean %.3f, frac kf with >=1 FP %.3f'%(pk.mean(),(pk>0).mean()))
print('FP names',fpname.most_common(10)); print('mislabel',misname.most_common(10))
for b in range(4): print('miss bin',b,miss[b], miss[b][0]/max(1,miss[b][1]))
tot=[sum(miss[b][0] for b in range(4)),sum(miss[b][1] for b in range(4))]; print('miss all',tot, tot[0]/max(1,tot[1]))
print('per class', {k:(v[0],v[1]) for k,v in cmiss.items()})
a=stats['strict']+stats['lenient']+stats['mislabel']; print('p_conf (mislabel/assoc) %.3f n %d'%(stats['mislabel']/max(1,a),a))
sx=[];sz=[]
for gi,L in lat.items():
    if len(L)<3: continue
    L=np.array(L); r=L-np.median(L,0); sx+=list(r[:,0])+list(r[:,1]); sz+=list(r[:,2])
sx=np.array(sx); sz=np.array(sz)
if len(sx): print('lat robust sigma xy %.3f (n %d) z %.3f'%(1.4826*np.median(np.abs(sx)),len(sx)//2,1.4826*np.median(np.abs(sz))))

import collections as C
for nm in sorted({v[1] for v in VIS}):
    L=[v for v in VIS if v[1]==nm]; print(nm,len(L),'dist %.2f-%.2f'%(min(v[2] for v in L),max(v[2] for v in L)),'miss',sum(v[3] for v in L))
dd=[]
for k,d in enumerate(I):
    p,Ro=cam(d['st'])
    for o in gt:
        if o['category'] in VOCAB_GT: dd.append((o['name'],round(float(np.linalg.norm(o['c']-p)),1)))
import itertools
for nm in sorted({x[0] for x in dd}): L=[x[1] for x in dd if x[0]==nm]; print('  all frames',nm,min(L),max(L))
