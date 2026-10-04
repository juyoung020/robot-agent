"""손가락 안쪽 틈(mm)을 그리퍼 각별·link5 x 구간별로 잰다(실제 메시, "hull" 인자면 시뮬 충돌 모양 = 볼록 껍질). E0."""
import trimesh,numpy as np
D='/home/juyoung/ra_ws/install/open_manipulator_description/share/open_manipulator_description/meshes/omx_f/'
f6=trimesh.load(D+'follower_07_gripper_motorized.stl'); f6.apply_scale(0.001)
import sys
if "hull" in sys.argv: f6=f6.convex_hull
f7=trimesh.load(D+'follower_08_gripper_gear.stl'); f7.apply_scale(0.001)
if "hull" in sys.argv: f7=f7.convex_hull
def pts(m,q,off):
    # sample surface densely
    p,_=trimesh.sample.sample_surface(m,200000,seed=0)
    p=np.vstack([p,m.vertices])
    c,s=np.cos(q),np.sin(q)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return p@R.T+np.array(off)
for qdeg in [0,5,10,20,30,45,60,75,90,100]:
    q=np.radians(qdeg)
    a=pts(f6,q,[0.0295,0.0075,0]); b=pts(f7,-q,[0.0295,-0.0108,0])
    rows=[]
    for x0 in np.arange(0.05,0.10,0.005):
        A=a[(a[:,0]>=x0)&(a[:,0]<x0+0.005)]; B=b[(b[:,0]>=x0)&(b[:,0]<x0+0.005)]
        if len(A)==0 or len(B)==0: rows.append((round(x0,3),None)); continue
        rows.append((round(x0,3),round((A[:,1].min()-B[:,1].max())*1000,1)))
    tip6=a[:,0].max(); tip7=b[:,0].max()
    print(qdeg,'tipx %.4f %.4f'%(tip6,tip7),rows)
