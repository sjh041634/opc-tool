"""Run: streamlit run app.py"""
import json, io, zipfile, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from opc_core import DEFAULT, GEOMETRY, run_cell, figure, save_result, expose, metrics, validate
st.set_page_config(page_title='OPC 도구',layout='wide')
st.title('INV_X1 / NAND2_X4 OPC 실험')
st.warning('내부 경계는 PDF 그림을 판독한 잠정 재구성 좌표입니다. 원 도면과 대조 후 geometry.json에서 수정하세요. 실제 공정 시뮬레이터가 아닙니다.')
c=DEFAULT.copy()
with st.sidebar:
 name=st.selectbox('레이아웃',['INV_X1','NAND2_X4'])
 c['sigma_nm']=st.number_input('가우시안 σ (nm)',1.75,7.0,3.5,1.75)
 c['threshold']=st.slider('현상 임계값 (상대 광량)',0.1,0.9,0.5,0.01)
 c['sweeps']=st.slider('최대 sweep',1,30,10)
 c['padding_nm']=max(21.0, c['kernel_sigma_radius']*c['sigma_nm']+c['search_radius_nm']) # default matches CLI
 st.caption('격자 1.75 nm, 노광 배율 1, 커널 ±ceil(3σ/Δx), 음성 레지스트 잔류 영역=1. 보정 전후 조건 고정.')
 geo=st.file_uploader('geometry.json 교체',type=['json'])
 default_path=Path(__file__).parent/'geometry.json'
 g=json.loads(geo.getvalue().decode('utf-8')) if geo else json.loads(default_path.read_text(encoding='utf-8')) if default_path.exists() else GEOMETRY
 st.download_button('현재 좌표 JSON',json.dumps(g,ensure_ascii=False,indent=2),'geometry.json')
signature=json.dumps([name,c,g],sort_keys=True)
if st.button('노광 계산 + 자동 OPC 실행',type='primary'):
 try:
  with st.spinner('두 단계 목적함수로 픽셀 마스크 최적화 중…'):
   r=run_cell(name,g,c)
  st.session_state['run']=(signature,r)
 except Exception as e:st.error(str(e))
if 'run' in st.session_state and st.session_state['run'][0]==signature:
 r=st.session_state['run'][1];s=r['summary']
 st.write('보정 전',s['before']);st.write('보정 후',s['after'])
 st.pyplot(figure(r,c,name))
 st.subheader('선택 사항: 마스크에 직사각형을 수동 추가·제거')
 st.caption('좌표는 여백을 포함한 배열 인덱스입니다. 행은 아래 방향, 열은 오른쪽 방향. 끝 인덱스 미포함. 적용 결과가 누적되며 다시 실행하면 초기화됩니다.')
 ny,nx=r['m'].shape
 cols=st.columns(4)
 y0=cols[0].number_input('시작 행',0,ny-1,0);y1=cols[1].number_input('끝 행',1,ny,1)
 x0=cols[2].number_input('시작 열',0,nx-1,0);x1=cols[3].number_input('끝 열',1,nx,1)
 value=st.radio('변경값',['투과=1','차광=0'],horizontal=True)
 if st.button('이 마스크에 적용하고 재계산'):
  if y1<=y0 or x1<=x0:st.error('끝 인덱스는 시작보다 커야 합니다.')
  else:
   r['m'][y0:y1,x0:x1]=value=='투과=1';r['ea']=expose(r['m'],r['k'],c)
   ss=r['summary'];ss['after']=metrics(r['t'],r['ea']>=c['threshold'],c)
   ss['delta_iou']=ss['after']['iou']-ss['before']['iou'];ss['area_reduction_percent']=100*(ss['before']['A_err_nm2']-ss['after']['A_err_nm2'])/ss['before']['A_err_nm2'] if ss['before']['A_err_nm2'] else None
   ss['manual_edits']=ss.get('manual_edits',[])+[dict(row_start=y0,row_stop=y1,col_start=x0,col_stop=x1,value=int(value=='투과=1'))]
   ss['final_mask_changed_pixels']=int((r['m']^r['t']).sum())
   st.session_state['run']=(signature,r);st.rerun()
 with tempfile.TemporaryDirectory() as tmp:
  save_result(r,c,name,tmp)
  Path(tmp,'geometry_used.json').write_text(json.dumps(g,ensure_ascii=False,indent=2),encoding='utf-8')
  Path(tmp,'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
   for p in Path(tmp).iterdir():z.write(p,p.name)
 st.download_button('현재 결과 전체 ZIP 저장',buf.getvalue(),name+'_results.zip','application/zip')
 st.caption('history/edits CSV는 자동 보정 기록입니다. 수동 수정 기록은 summary.json의 manual_edits에 별도 기록됩니다.')
else:st.info('조건을 선택하고 실행 버튼을 누르세요.')
