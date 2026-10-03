// 분석용(학습 경로 아님): map.h MP 의 표류 + keyframe 보정(phase_begin / phase_objects 3a 와 같은 식)을 정해진 궤적에서 돌려
// 자세 오차 통계(rms·max, xy·yaw)를 낸다. keyframe 맞추기는 늘 성공(맞은 열 ≥ min_hits)으로 둔다.
//   g++ -O2 -o mpdrift mpdrift.cpp
//   ./mpdrift <kind 0=gt_move 대본, 1=탐색 비슷> <odo_t> <odo_rr> <odo_rt> <kf_corr> [판 수=200]
//   DBG=1 이면 yaw 오차 > 0.08 rad 인 스텝을 찍는다.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
static uint64_t sm(uint64_t& s){s+=0x9E3779B97F4A7C15ull;uint64_t z=s;z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;return z^(z>>31);}
static float r01(uint64_t& s){return (float)(sm(s)>>40)*(1.0f/16777216.0f);}
static float gauss(uint64_t& s){float a=r01(s)+r01(s)+r01(s)+r01(s);return (a-2.f)*1.7320508f;}
static float wrap(float a){while(a>M_PI)a-=2*M_PI;while(a<-M_PI)a+=2*M_PI;return a;}
struct P{float odo_t,odo_rr,odo_rt,kf_corr;};
struct Cmd{float v,w;};
// G1 kinematics (env.h: dt 0.01, a_v 1, a_w 3), 10 Hz control
struct Traj{std::vector<float> x,y,yaw,v,w;};
static void sim_ctrl(Traj& T,float& x,float& y,float& th,float& v,float& w,float vc,float wc){
  for(int k=0;k<10;++k){float dv=vc-v,dw=wc-w;v+=fmaxf(-0.01f,fminf(0.01f,dv));w+=fmaxf(-0.03f,fminf(0.03f,dw));
    float h=th+0.5f*w*0.01f;x+=v*cosf(h)*0.01f;y+=v*sinf(h)*0.01f;th=wrap(th+w*0.01f);}
  T.x.push_back(x);T.y.push_back(y);T.yaw.push_back(th);T.v.push_back(v);T.w.push_back(w);}
static void turn(Traj& T,float& x,float& y,float& th,float& v,float& w,float ang,float wm){
  float done=0;while(fabsf(done)<fabsf(ang)-0.01f){float rem=fabsf(ang)-fabsf(done);float wc=fminf(wm,fmaxf(0.1f,sqrtf(2*3.f*rem)))*(ang>0?1:-1);
    float p=th;sim_ctrl(T,x,y,th,v,w,0,wc);done+=wrap(th-p);if(T.x.size()>100000)break;}
  for(int i=0;i<3;++i)sim_ctrl(T,x,y,th,v,w,0,0);}
static void drive(Traj& T,float& x,float& y,float& th,float& v,float& w,float d,float vm,float curv){
  float s=0;while(s<d-0.005f){float rem=d-s;float vc=fminf(vm,fmaxf(0.05f,sqrtf(2*1.f*rem)));float px=x,py=y;sim_ctrl(T,x,y,th,v,w,vc,vc*curv);s+=hypotf(x-px,y-py);}
  for(int i=0;i<3;++i)sim_ctrl(T,x,y,th,v,w,0,0);}
static Traj make(int kind,uint64_t& rng){
  Traj T;float x=0,y=0,th=0,v=0,w=0;
  if(kind==0){ // gt_move script (R1 run): ~0.6 rad/s turns, ~0.27 m/s drives, pauses between calls
    float seq[][2]={{0,90},{0,90},{0,90},{0,90},{1,0},{0,180},{1,0},{0,-90},{0.8f,0},{0,180},{0.8f,0},{0,90}};
    for(int i=0;i<25;++i)sim_ctrl(T,x,y,th,v,w,0,0);
    for(auto&s:seq){if(s[0]>0)drive(T,x,y,th,v,w,s[0],0.27f,0);else turn(T,x,y,th,v,w,s[1]*M_PI/180,0.6f);for(int i=0;i<10;++i)sim_ctrl(T,x,y,th,v,w,0,0);}
    while(T.x.size()<890)sim_ctrl(T,x,y,th,v,w,0,0);
  } else { // explore-like (go_to frontiers): turn toward goal, drive with mild curvature, ~16.5 m / 68 s
    float L=0;while(L<16.5f){float a=(r01(rng)*2-1)*2.6f;turn(T,x,y,th,v,w,a,0.8f);float d=0.5f+3.f*r01(rng);if(L+d>16.5f)d=16.5f-L+0.01f;
      drive(T,x,y,th,v,w,d,0.45f,(r01(rng)*2-1)*0.4f);L+=d;}
  }
  return T;}
struct Stat{double rms_xy,max_xy,rms_yaw,max_yaw,L,turn,T;};
static Stat run(const Traj& T,const P& p,uint64_t seed){
  uint64_t r=seed;float ex=T.x[0],ey=T.y[0],eyaw=T.yaw[0],px=ex,py=ey,pyaw=eyaw,lx=ex,ly=ey,lyaw=eyaw,vmax=0,wmax=0;int since=0;bool first=true;
  double s2=0,s2y=0,mx=0,my=0,L=0,tu=0;int n=0;
  for(size_t i=0;i<T.x.size();++i){
    float x=T.x[i],y=T.y[i],yaw=T.yaw[i];
    if(i>0){float s0=sinf(pyaw),c0=cosf(pyaw),dxw=x-px,dyw=y-py,dxb=c0*dxw+s0*dyw,dyb=-s0*dxw+c0*dyw,dth=wrap(yaw-pyaw),dist=sqrtf(dxb*dxb+dyb*dyb);
      L+=dist;tu+=fabsf(dth);
      float n1=gauss(r),n2=gauss(r),n3=gauss(r),sx=p.odo_t*dist,nxb=dxb+n1*sx,nyb=dyb+n2*sx,nth=dth+n3*(p.odo_rr*fabsf(dth)+p.odo_rt*dist);
      float se=sinf(eyaw),ce=cosf(eyaw);ex+=ce*nxb-se*nyb;ey+=se*nxb+ce*nyb;eyaw=wrap(eyaw+nth);}
    px=x;py=y;pyaw=yaw;vmax=fmaxf(vmax,fabsf(T.v[i]));wmax=fmaxf(wmax,fabsf(T.w[i]));since++;
    float dx=ex-lx,dy=ey-ly;bool moved=dx*dx+dy*dy>=0.05f*0.05f||fabsf(wrap(eyaw-lyaw))>=0.034906585f;
    bool kf=first||moved||since>=50;
    if(kf){bool still=!first&&vmax<0.01f&&wmax<0.01f;
      if(!first&&!still){float keep=1-p.kf_corr;ex=x+(ex-x)*keep;ey=y+(ey-y)*keep;eyaw=wrap(yaw+wrap(eyaw-yaw)*keep);}
      lx=ex;ly=ey;lyaw=eyaw;since=0;vmax=0;wmax=0;first=false;}
    double exy=hypot(ex-x,ey-y),ey_=fabs(wrap(eyaw-yaw)); if(getenv("DBG")&&ey_>0.08)printf("i=%zu kf=%d v=%.3f w=%.3f err=%.3f\n",i,(int)kf,T.v[i],T.w[i],ey_);s2+=exy*exy;s2y+=ey_*ey_;mx=fmax(mx,exy);my=fmax(my,ey_);n++;}
  return {sqrt(s2/n),mx,sqrt(s2y/n),my,L,tu,T.x.size()*0.1};}
int main(int argc,char**argv){
  int kind=atoi(argv[1]);P p{(float)atof(argv[2]),(float)atof(argv[3]),(float)atof(argv[4]),(float)atof(argv[5])};int N=argc>6?atoi(argv[6]):200;
  double a[4]={0,0,0,0},L=0,tu=0,TT=0;
  for(int k=0;k<N;++k){uint64_t g=1000+k;Traj T=make(kind,g);Stat s=run(T,p,77+k*7919);a[0]+=s.rms_xy;a[1]+=s.max_xy;a[2]+=s.rms_yaw;a[3]+=s.max_yaw;L+=s.L;tu+=s.turn;TT+=s.T;}
  printf("kind %d odo_t %.4f odo_rr %.4f odo_rt %.4f kf_corr %.3f | path %.2f m turn %.0f deg T %.1f s | rms_xy %.2f cm max_xy %.2f cm rms_yaw %.3f deg max_yaw %.3f deg\n",
    kind,p.odo_t,p.odo_rr,p.odo_rt,p.kf_corr,L/N,tu/N*57.2958,TT/N,a[0]/N*100,a[1]/N*100,a[2]/N*57.2958,a[3]/N*57.2958);}
