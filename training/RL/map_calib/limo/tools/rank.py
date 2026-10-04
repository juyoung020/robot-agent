import re,sys,math
T2=(3.05,0.256,5.34,0.662); T3=(1.32,0.206,2.72,0.372)
rows=[]
for l in open(sys.argv[1]):
    p=l.split('|')
    if len(p)<4: continue
    par=p[0].split()
    def g(s):
        m=re.search(r'rms ([\d.]+) cm / ([\d.]+) deg, max ([\d.]+) cm / ([\d.]+) deg',s); return tuple(map(float,m.groups()))
    try: a=g(l.split('map_drift CPU kind 2')[1]); b=g(l.split('map_drift CPU kind 3')[1])
    except Exception: continue
    def loss(x,T): return sum(w*math.log(x[i]/T[i])**2 for i,w in enumerate((1,1,0.5,0.5)))
    rows.append((loss(a,T2),loss(b,T3),par,a,b))
rows.sort(key=lambda r:r[0])
for r in rows[:int(sys.argv[2]) if len(sys.argv)>2 else 15]: print('%.3f %.3f'%(r[0],r[1]),' '.join(r[2]),'| k2',r[3],'| k3',r[4])
print('n',len(rows))
