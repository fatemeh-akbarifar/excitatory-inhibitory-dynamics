"""Two-population neural dynamics; dimensionless teaching model, not fitted recordings."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def rhs(t,y,P=1.,w=10.,coupled=True):
 e,i=y
 return np.array([-e+expit((12*e-w*i if coupled else 0)+P-2),(-i+expit((10*e if coupled else 0)-3))/2])
def jac(y,P=1.,w=10.):
 e,i=y;a=expit(12*e-w*i+P-2);b=expit(10*e-3)
 return np.array([[-1+12*a*(1-a),-w*a*(1-a)],[5*b*(1-b),-.5]])
def simulate(P=1.,w=10.,initial=(.1,.1),duration=200.,step=.05,coupled=True):
 t=np.linspace(0,duration,round(duration/.05)+1)
 sol=solve_ivp(lambda t,y:rhs(t,y,P,w,coupled),(0,duration),initial,t_eval=t,rtol=1e-8,atol=1e-10,max_step=step)
 if not sol.success or not np.isfinite(sol.y).all() or sol.y.min() < -1e-8 or sol.y.max()>1+1e-8:raise RuntimeError('Invalid population trajectory')
 return sol.t,sol.y.T

def describe(t,y,P,w):
 z=y[t>=t[-1]/2];tt=t[t>=t[-1]/2];amp=np.ptp(z,axis=0);peaks,_=find_peaks(z[:,0],prominence=.001)
 gaps=np.diff(tt[peaks]);cv=np.std(gaps)/np.mean(gaps) if len(gaps)>1 else np.nan
 halves=[np.ptp(v[:,0]) for v in np.array_split(z,2)];amp_drift=abs(halves[1]-halves[0])/max(halves+[1e-12])
 label='unresolved'
 if max(amp)<.001 and np.linalg.norm(rhs(0,z[-1],P,w))<1e-4:label='near-steady'
 elif len(peaks)>=5 and cv<.05 and amp_drift<.05:label='apparently-periodic'
 return {'mean_E':z[:,0].mean(),'min_E':z[:,0].min(),'max_E':z[:,0].max(),'amplitude_E':amp[0],'amplitude_I':amp[1],'period':np.mean(gaps) if len(gaps) else np.nan,'peak_count':len(peaks),'interval_cv':cv,'amplitude_drift':amp_drift,'behaviour':label}

def equilibria(P,w):
 found=[]
 for a in np.linspace(0,1,5):
  for b in np.linspace(0,1,5):
   result=root(lambda y:rhs(0,y,P,w),[a,b],jac=lambda y:jac(y,P,w));x=result.x
   if np.linalg.norm(rhs(0,x,P,w))<1e-7 and np.min(x)>=0 and np.max(x)<=1 and not any(np.linalg.norm(x-z)<1e-5 for z in found):found.append(x)
 return found

def checks():
 t,y=simulate(coupled=False);target=np.array([expit(-1),expit(-3)]);exact=target+(.1-target)*np.exp(-t[:,None]/np.array([1,2]))
 error=float(abs(y-exact).max());assert error<1e-7
 x=np.array([.3,.4]);eps=1e-6;fd=np.column_stack([(rhs(0,x+eps*np.eye(2)[k])-rhs(0,x-eps*np.eye(2)[k]))/(2*eps) for k in range(2)])
 jerr=float(abs(jac(x)-fd).max());assert jerr<1e-7
 records=[]
 for P in (0.,1.,4.):
  t,y=simulate(P);_,fine=simulate(P,step=.025);a=describe(t,y,P,10);b=describe(t,fine,P,10)
  assert abs(a['mean_E']-b['mean_E'])<1e-4
  assert abs(a['amplitude_E']-b['amplitude_E'])<max(1e-4,.01*b['amplitude_E'])
  if np.isfinite(a['period']):assert abs(a['period']-b['period'])<.01*b['period']
  records.append({'P':P,'maximum_trajectory_difference':float(abs(y-fine).max())})
 return {'uncoupled_error':error,'jacobian_error':jerr,'refinement':records}

def main():
 OUT.mkdir(exist_ok=True);validation=checks()
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors=['#26828e','#a34e91'];traces={};eqrows=[]
 fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
 fig2,phase=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
 grid=np.linspace(0,1,31);xx,yy=np.meshgrid(grid,grid)
 for ax,pa,P in zip(axes,phase,(0.,1.,4.)):
  t,y=simulate(P);traces[P]=(t,y)
  for k,c in enumerate(colors):ax.plot(t,y[:,k],color=c,label=['Excitatory','Inhibitory'][k])
  ax.set(title=f'Input P = {P:g}',xlabel='Model time',ylim=(0,1));ax.legend(frameon=False)
  field=rhs(0,[xx,yy],P);pa.streamplot(xx,yy,field[0],field[1],density=.6,color='#c8cdd2',linewidth=.6)
  pa.contour(xx,yy,field[0],levels=[0],colors=[colors[0]]);pa.contour(xx,yy,field[1],levels=[0],colors=[colors[1]])
  pa.plot(y[:,0],y[:,1],color='#333333',lw=1);pa.set(title=f'Input P = {P:g}',xlabel='E',xlim=(0,1),ylim=(0,1))
  for eq in equilibria(P,10):
   ev=np.linalg.eigvals(jac(eq,P));stable=bool(max(ev.real)<0)
   pa.scatter(*eq,color='#111111',marker='o' if stable else 'x',s=35)
   eqrows.append({'P':P,'E':eq[0],'I':eq[1],'max_real_eigenvalue':max(ev.real),'eigenvalue_1_real':ev[0].real,'eigenvalue_1_imag':ev[0].imag,'eigenvalue_2_real':ev[1].real,'eigenvalue_2_imag':ev[1].imag,'locally_stable':stable})
 axes[0].set_ylabel('Population activity');phase[0].set_ylabel('I')
 fig.tight_layout();fig.savefig(OUT/'activity.png',dpi=170);plt.close(fig)
 fig2.suptitle('Nullclines, flow and the observed trajectory');fig2.tight_layout();fig2.savefig(OUT/'phase-portraits.png',dpi=170);plt.close(fig2)
 pd.DataFrame(eqrows).to_csv(OUT/'equilibria.csv',index=False)
 pd.DataFrame([{'P':P,'time':a,'E':b,'I':c} for P,(t,y) in traces.items() for a,(b,c) in zip(t,y)]).to_csv(OUT/'representative-trajectories.csv',index=False)
 rows=[]
 for w in (8.,10.,12.):
  for P in np.linspace(0,4,21):
   for init in (.05,.5,.9):
    t,y=simulate(P,w,(init,init));m=describe(t,y,P,w);extended=False
    if m['behaviour']=='unresolved':
     t,y=simulate(P,w,(init,init),duration=400);m=describe(t,y,P,w);extended=True
    rows.append({'P':P,'w_EI':w,'initial_E_I':init,'extended_to_400':extended,**m})
  print('Completed inhibitory weight',w,flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'parameter-sweep.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4))
 for init in (.05,.5,.9):
  g=df[(df.w_EI==10)&(df.initial_E_I==init)];ax.plot(g.P,g.min_E,label=f'Initial E=I={init}',lw=1.2);ax.plot(g.P,g.max_E,lw=1.2)
 ax.set(xlabel='Input P',ylabel='Late minimum / maximum E',title='Input changes the range of activity');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'input-response.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
 for ax,init in zip(axes,(.05,.5,.9)):
  a=df[df.initial_E_I==init].pivot(index='w_EI',columns='P',values='amplitude_E');im=ax.imshow(a,origin='lower',aspect='auto',extent=(-.1,4.1,7,13),vmin=0,vmax=1,cmap='magma');ax.set(xlabel='Input P',title=f'Initial E=I={init}',yticks=[8,10,12])
 axes[0].set_ylabel('Inhibitory weight');fig.colorbar(im,ax=axes,label='Late E amplitude',shrink=.8);fig.savefig(OUT/'feedback-map.png',dpi=170,bbox_inches='tight');plt.close(fig)
 perturb=[];fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,P in zip(axes,(0.,1.)):
  t,y=traces[P];state=y[np.argmin(abs(t-100))].copy();state[0]+=.02
  tt,shift=simulate(P,initial=state,duration=100);base=y[t>=100];dist=np.linalg.norm(shift-base,axis=1);recovery=None
  if P==0:
   for k in range(len(tt)-200):
    if np.all(dist[k:k+201]<.0002):recovery=float(tt[k]);break
  ax.plot(tt,base[:,0],label='Original',color=colors[0]);ax.plot(tt,shift[:,0],label='Perturbed E + 0.02',color=colors[1],ls='--');ax.set(title=f'Input P={P:g}',xlabel='Time after perturbation',ylabel='E');ax.legend(fontsize=8)
  perturb.append({'P':P,'recovery_time':recovery,'original_amplitude':float(np.ptp(base[-1000:,0])),'perturbed_amplitude':float(np.ptp(shift[-1000:,0]))})
 fig.tight_layout();fig.savefig(OUT/'perturbation.png',dpi=170);plt.close(fig)
 pd.DataFrame(perturb).to_csv(OUT/'perturbations.csv',index=False)
 summary={'checks':validation,'conditions':len(df),'behaviour_counts':df.behaviour.value_counts().to_dict(),'perturbations':perturb,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'source_sha256':hashlib.sha256((ROOT/'analysis.py').read_bytes()).hexdigest(),'data':'Simulated; model time is dimensionless'}
 (OUT/'run.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return df
if __name__=='__main__':main()
