"""Educational OPC: relative intensity convolution, not coherent optical simulation."""
from pathlib import Path
import json, csv, argparse, platform
import numpy as np
import scipy
from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation, binary_erosion, label

DEFAULT = dict(dx_nm=1.75, sigma_nm=3.5, threshold=0.5, dose=1.0, kernel_sigma_radius=3.0, padding_nm=21.0, sweeps=10, seed=2026, search_radius_nm=7.0)
GEOMETRY = {
 'status':'PROVISIONAL: internal edges estimated from assignment Fig.3; inspect against source before submission',
 'units':'nm; local origin top-left; x right, y down; half-open rectangles [x0,x1) x [y0,y1)',
 'INV_X1':dict(width_nm=63.0,height_nm=189.0,rectangles=[[0,0,63,10.5],[0,178.5,63,189],[10.5,10.5,21,73.5],[10.5,115.5,21,178.5],[42,63,52.5,126]]),
 'NAND2_X4':dict(width_nm=157.5,height_nm=189.0,rectangles=[[0,0,157.5,10.5],[0,178.5,157.5,189],[10.5,10.5,21,42],[73.5,10.5,84,42],[136.5,10.5,147,42],[42,21,52.5,84],[105,21,115.5,84],[42,73.5,115.5,84],[10.5,94.5,21,168],[42,94.5,52.5,168],[73.5,84,84,168],[105,94.5,115.5,168],[136.5,94.5,147,178.5]])}

def validate(c):
 for k in ['dx_nm','sigma_nm','dose','kernel_sigma_radius']:
  if c[k]<=0: raise ValueError(k+' must be positive')
 if not 0<c['threshold']<c['dose']: raise ValueError('Require 0 < threshold < dose')
 if c['padding_nm']< c['kernel_sigma_radius']*c['sigma_nm']+c['search_radius_nm']: raise ValueError('Padding must cover kernel radius + mask search radius')
 if c['sweeps']<0 or c['search_radius_nm']<0: raise ValueError('Invalid search bounds')

def kernel(c):
 r=int(np.ceil(c['kernel_sigma_radius']*c['sigma_nm']/c['dx_nm']))
 y,x=np.mgrid[-r:r+1,-r:r+1]*c['dx_nm']
 k=np.exp(-(x*x+y*y)/(2*c['sigma_nm']**2)); return k/k.sum()

def target(cell,c):
 dx=c['dx_nm']; pad=int(np.ceil(c['padding_nm']/dx))
 nx=int(np.ceil(cell['width_nm']/dx))+2*pad; ny=int(np.ceil(cell['height_nm']/dx))+2*pad
 x=(np.arange(nx)+0.5-pad)*dx; y=(np.arange(ny)+0.5-pad)*dx
 t=np.zeros((ny,nx),bool)
 for x0,y0,x1,y1 in cell['rectangles']:
  if not(0<=x0<x1<=cell['width_nm'] and 0<=y0<y1<=cell['height_nm']): raise ValueError('Rectangle outside layout bounds')
  t |= (y[:,None]>=y0)&(y[:,None]<y1)&(x[None,:]>=x0)&(x[None,:]<x1)
 return t,pad

def expose(m,k,c): return c['dose']*convolve2d(m.astype(float),k,mode='same',boundary='fill',fillvalue=0)

def metrics(t,p,c):
 inter=int((t&p).sum()); union=int((t|p).sum()); err=int((t^p).sum())
 return dict(iou=inter/union if union else 1.0,error_pixels=err,A_err_nm2=err*c['dx_nm']**2,A_target_nm2=int(t.sum())*c['dx_nm']**2,normalized_error=err/int(t.sum()) if t.any() else None,missing_pixels=int((t&~p).sum()),excess_pixels=int((~t&p).sum()),components_4connected=int(label(p)[1]))

def optimize(t,k,c):
 """Sequential binary pixel coordinate descent; accept lexicographic (XOR pixels, intensity MSE) improvements. Not named serif/edge parameter optimization."""
 m=t.copy(); e=expose(m,k,c); r=k.shape[0]//2; dk=c['dose']*k
 allowed=binary_dilation(t,iterations=max(1,int(np.ceil(c['search_radius_nm']/c['dx_nm'])))) if c['search_radius_nm']>0 else t.copy()
 allowed[:r]=False; allowed[-r:]=False; allowed[:,:r]=False; allowed[:,-r:]=False
 rng=np.random.default_rng(c['seed']); history=[]; edits=[]
 def record(s,accepted):
  z=metrics(t,e>=c['threshold'],c); z.update(sweep=s,accepted_flips=accepted,intensity_mse=float(np.mean((e/c['dose']-t)**2)));history.append(z)
 record(0,0)
 for sweep in range(1,c['sweeps']+1):
  boundary=binary_dilation(m)^binary_erosion(m)
  coords=np.argwhere(boundary&allowed); rng.shuffle(coords); accepted=0
  for y,x in coords:
   sy=slice(y-r,y+r+1); sx=slice(x-r,x+r+1)
   old=e[sy,sx]; tt=t[sy,sx]; sign=-1 if m[y,x] else 1; new=old+sign*dk
   gain=int(np.count_nonzero((old>=c['threshold'])!=tt))-int(np.count_nonzero((new>=c['threshold'])!=tt))
   mse_delta=float(np.sum((new/c['dose']-tt)**2-(old/c['dose']-tt)**2))
   if gain>0 or (gain==0 and mse_delta < -1e-12):
    m[y,x]=not m[y,x]; e[sy,sx]=new; accepted+=1
    edits.append(dict(sweep=sweep,row=int(y),col=int(x),new_value=int(m[y,x]),error_pixel_reduction=gain))
  record(sweep,accepted)
  if not accepted: break
 # verify incremental convolution and threshold consistency
 exact=expose(m,k,c)
 if not np.allclose(e,exact,atol=1e-12): raise AssertionError('Incremental convolution mismatch')
 # Near-threshold floating ambiguity is explicitly checked.
 if np.any((e>=c['threshold'])!=(exact>=c['threshold'])): raise AssertionError('Threshold ambiguity; change parameters or use exact evaluation')
 return m,exact,history,edits

def run_cell(name,g,c):
 validate(c); t,pad=target(g[name],c); k=kernel(c); eb=expose(t,k,c)
 m,ea,h,edits=optimize(t,k,c)
 before=metrics(t,eb>=c['threshold'],c); after=metrics(t,ea>=c['threshold'],c)
 summary=dict(cell=name,geometry_status=g['status'],config=c,array_shape_yx=list(t.shape),domain_nm_xy=[t.shape[1]*c['dx_nm'],t.shape[0]*c['dx_nm']],actual_padding_nm=pad*c['dx_nm'],kernel_shape=list(k.shape),kernel_sum=float(k.sum()),kernel_radius_nm=(k.shape[0]//2)*c['dx_nm'],target_components_4connected=int(label(t)[1]),before=before,after=after,delta_iou=after['iou']-before['iou'],area_reduction_percent=100*(before['A_err_nm2']-after['A_err_nm2'])/before['A_err_nm2'] if before['A_err_nm2'] else None,completed_sweeps=len(h)-1,accepted_pixel_flips=len(edits),final_mask_changed_pixels=int((t^m).sum()))
 return dict(t=t,m=m,eb=eb,ea=ea,k=k,pad=pad,history=h,edits=edits,summary=summary)

def figure(result,c,name):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 from matplotlib.patches import Patch
 t=result['t']; m=result['m']; eb=result['eb']; ea=result['ea']; pb=eb>=c['threshold']; pa=ea>=c['threshold']
 def overlay(p): return np.where(t&p,1,np.where(t&~p,2,np.where(~t&p,3,0)))
 cmap=ListedColormap(['white','#2563eb','#ef4444','#f59e0b'])
 panels=[(t,'Target',False),(t,'Before mask',False),(m,'After mask',False),(eb,'Before exposure',True),(pb,'Before developed',False),(overlay(pb),'Before errors',False),(ea,'After exposure',True),(pa,'After developed',False),(overlay(pa),'After errors',False)]
 fig,ax=plt.subplots(3,3,figsize=(12,13),layout='constrained')
 pad=result['pad']*c['dx_nm']; extent=[-pad,t.shape[1]*c['dx_nm']-pad,t.shape[0]*c['dx_nm']-pad,-pad]
 for a,(data,title,is_e) in zip(ax.flat,panels):
  is_o='errors' in title
  im=a.imshow(data,origin='upper',extent=extent,interpolation='nearest',cmap='viridis' if is_e else cmap if is_o else 'Blues',vmin=0,vmax=c['dose'] if is_e else 3 if is_o else 1)
  a.set_title(title);a.set_xlabel('x (nm)');a.set_ylabel('y (nm)');a.set_aspect('equal')
  if is_e: fig.colorbar(im,ax=a,label='Relative exposure')
 fig.legend(handles=[Patch(color='#2563eb',label='Match'),Patch(color='#ef4444',label='Missing'),Patch(color='#f59e0b',label='Excess')],loc='outside lower center',ncol=3)
 fig.suptitle(name+' | provisional reconstructed geometry | educational model')
 return fig

def save_result(result,c,name,out):
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 (out/(name+'_summary.json')).write_text(json.dumps(result['summary'],ensure_ascii=False,indent=2),encoding='utf-8')
 np.savez_compressed(out/(name+'_arrays.npz'),target=result['t'],mask_before=result['t'],mask_after=result['m'],exposure_before=result['eb'],exposure_after=result['ea'],developed_before=result['eb']>=c['threshold'],developed_after=result['ea']>=c['threshold'],kernel=result['k'])
 for key in ['t','m','eb','ea']:
  np.savetxt(out/(name+'_'+key+'.csv'),result[key],delimiter=',',fmt='%.12g')
 for key in ['history','edits']:
  rows=result[key]
  if rows:
   with (out/(name+'_'+key+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig=figure(result,c,name);fig.savefig(out/(name+'_comparison.png'),dpi=180);plt.close(fig)

def self_tests(c):
 k=kernel(c);z=np.zeros((51,51),bool); one=z.copy();one[25,25]=True
 e=expose(one,k,c);t,_=target(GEOMETRY['INV_X1'],c)
 tests=dict(kernel_normalized=bool(abs(k.sum()-1)<1e-12),kernel_symmetric=bool(np.allclose(k,k[::-1,::-1])),empty_mask_zero=bool(np.all(expose(z,k,c)==0)),impulse_response=bool(np.allclose(e[25-k.shape[0]//2:26+k.shape[0]//2,25-k.shape[1]//2:26+k.shape[1]//2],c['dose']*k)),identical_iou=metrics(t,t,c)['iou']==1,identical_error=metrics(t,t,c)['A_err_nm2']==0,six_grid_line_width=abs(10.5/c['dx_nm']-6)<1e-12)
 assert all(tests.values()),tests
 return tests

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--geometry');p.add_argument('--config');a=p.parse_args()
 c=DEFAULT.copy()
 if a.config:c.update(json.loads(Path(a.config).read_text(encoding='utf-8')))
 g=json.loads(Path(a.geometry).read_text(encoding='utf-8')) if a.geometry else GEOMETRY
 validate(c);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8');(out/'geometry_used.json').write_text(json.dumps(g,indent=2,ensure_ascii=False),encoding='utf-8')
 tests=self_tests(c); summaries=[]
 for name in ['INV_X1','NAND2_X4']:
  r=run_cell(name,g,c);r2=run_cell(name,g,c)
  tests[name+'_deterministic_rerun']=bool(np.array_equal(r['m'],r2['m']) and np.array_equal(r['ea'],r2['ea']))
  tests[name+'_nonincreasing_error']=all(b['error_pixels']<=aa['error_pixels'] for aa,b in zip(r['history'],r['history'][1:]))
  tests[name+'_exact_convolution']=True
  save_result(r,c,name,out);summaries.append(r['summary']);print(json.dumps(r['summary'],ensure_ascii=False))
 assert all(tests.values())
 versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
 (out/'verification.json').write_text(json.dumps(dict(tests=tests,versions=versions),indent=2),encoding='utf-8')
 lines=['# 실제 실행 결과 (교육용 재구성 형상)','',g['status'],'','방법: 이진 픽셀 좌표 하강법. 오차 픽셀 수를 우선 최소화하고 동률일 때 광량 MSE를 줄임. 경계 이동량·선 끝 확장량·모서리 크기를 별도 최적화한 알고리즘은 아님.','',json.dumps(c,ensure_ascii=False),'']
 for s in summaries:
  lines += [f"## {s['cell']}",f"IoU: {s['before']['iou']:.8f} → {s['after']['iou']:.8f}",f"A_err: {s['before']['A_err_nm2']:.4f} → {s['after']['A_err_nm2']:.4f} nm²",f"ΔIoU: {s['delta_iou']:.8f}; 면적 감소율: {s['area_reduction_percent']:.4f}%",f"배열 (y,x): {s['array_shape_yx']}; 계산영역 (x,y) nm: {s['domain_nm_xy']}",f"실행 sweep: {s['completed_sweeps']}; 채택 픽셀 반전: {s['accepted_pixel_flips']}",f"4방향 연결성분 수 목표/보정 전/후: {s['target_components_4connected']}/{s['before']['components_4connected']}/{s['after']['components_4connected']}",'']
 lines += ['## 주의','연결성분 개수가 같다는 것만으로 배선의 연결/단절이 없다고 증명되지는 않음. 비교 그림과 좌표를 확인할 것.','좌표 재구성, σ·임계값은 교육용 설정이며 실제 공정 보정값이 아님. 원 도면 검증·실제 AI 사용 이력·사용자 재실행·온라인 배포는 별도로 필요.']
 (out/'실행결과_요약.md').write_text('\n\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()
