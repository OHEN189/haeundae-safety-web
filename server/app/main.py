import hmac,io,json,os,uuid
from datetime import datetime,timezone
from pathlib import Path
from fastapi import FastAPI,File,Form,Header,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image,UnidentifiedImageError
from .yolo import detect

ROOT=Path(__file__).resolve().parents[1]
UPLOADS=ROOT/'uploads'; REPORTS=ROOT/'data'/'reports.json';UPLOADS.mkdir(exist_ok=True)
with open(ROOT/'data'/'species.json',encoding='utf-8') as f: SPECIES={x['id']:x for x in json.load(f)}
app=FastAPI(title='Haeundae Marine Safety API')
ADMIN_PASSWORD=os.getenv('ADMIN_PASSWORD','1890')
DEFAULT_ORIGINS=','.join([
 'http://localhost:3000',
 'http://127.0.0.1:3000',
 'http://localhost:3001',
 'http://127.0.0.1:3001',
 'http://localhost:8080',
 'http://127.0.0.1:8080',
 'http://localhost:5500',
 'http://127.0.0.1:5500',
 'https://ohen189.github.io',
 'null',
])
allowed_origins=[
 origin.strip()
 for origin in os.getenv('ALLOWED_ORIGINS',DEFAULT_ORIGINS).split(',')
 if origin.strip()
]
app.add_middleware(
 CORSMiddleware,
 allow_origins=allowed_origins,
 allow_methods=['*'],
 allow_headers=['*'],
)
app.mount('/uploads',StaticFiles(directory=UPLOADS),name='uploads')
@app.get('/health')
def health(): return {'ok':True}
@app.get('/species')
def species(): return list(SPECIES.values())
@app.get('/reports/admin')
def admin_reports(x_admin_password:str=Header(default='')):
 if not hmac.compare_digest(x_admin_password,ADMIN_PASSWORD):
  raise HTTPException(401,'관리자 비밀번호가 올바르지 않습니다.')
 reports=json.loads(REPORTS.read_text(encoding='utf-8')) if REPORTS.exists() else []
 return sorted(reports,key=lambda report:report.get('discoveredAt',''),reverse=True)
@app.get('/reports/approved')
def approved_reports():
 reports=json.loads(REPORTS.read_text(encoding='utf-8')) if REPORTS.exists() else []
 return [
  report for report in reports
  if report.get('approved') is True
  and report.get('latitude') is not None
  and report.get('longitude') is not None
 ]
@app.post('/reports/{report_id}/review/{decision}')
def review_report(report_id:str,decision:str,x_admin_password:str=Header(default='')):
 if not hmac.compare_digest(x_admin_password,ADMIN_PASSWORD):
  raise HTTPException(401,'관리자 비밀번호가 올바르지 않습니다.')
 if decision not in {'approve','reject'}:
  raise HTTPException(400,'승인 또는 반려 중 하나를 선택해 주세요.')
 reports=json.loads(REPORTS.read_text(encoding='utf-8')) if REPORTS.exists() else []
 report=next((item for item in reports if item.get('id')==report_id),None)
 if report is None:
  raise HTTPException(404,'제보를 찾을 수 없습니다.')
 if decision=='approve' and (report.get('latitude') is None or report.get('longitude') is None):
  raise HTTPException(400,'좌표가 없는 기존 제보는 지도에 승인할 수 없습니다.')
 report['approved']=decision=='approve'
 report['reviewStatus']='approved' if decision=='approve' else 'rejected'
 report['reviewedAt']=datetime.now(timezone.utc).isoformat()
 REPORTS.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
 return {
  'id':report_id,
  'approved':report['approved'],
  'reviewStatus':report['reviewStatus'],
  'message':'제보를 승인해 지도에 반영했습니다.' if report['approved'] else '제보를 반려했습니다.',
 }
@app.post('/detect')
async def detection(file:UploadFile=File(...)):
 if file.content_type not in {'image/jpeg','image/png'}: raise HTTPException(400,'JPG, JPEG, PNG 파일만 업로드할 수 있습니다.')
 raw=await file.read()
 if len(raw)>10*1024*1024: raise HTTPException(400,'이미지 크기는 10MB 이하여야 합니다.')
 try: image=Image.open(io.BytesIO(raw)).convert('RGB')
 except UnidentifiedImageError: raise HTTPException(400,'열 수 없는 이미지 파일입니다.')
 try: detections=detect(image,SPECIES)
 except FileNotFoundError: raise HTTPException(503,'AI 모델 파일을 찾을 수 없습니다. server/models/best.pt를 확인하세요.')
 except Exception: raise HTTPException(500,'이미지 분석 중 오류가 발생했습니다. 잠시 후 다시 시도하세요.')
 return {'detections':detections,'imageWidth':image.width,'imageHeight':image.height}
@app.post('/reports')
async def create_report(location:str=Form(...),discovered_at:str=Form(...),species_type:str=Form(...),latitude:float=Form(...),longitude:float=Form(...),content:str=Form(''),file:UploadFile=File(...)):
 if file.content_type not in {'image/jpeg','image/png'}: raise HTTPException(400,'JPG, JPEG, PNG 파일만 업로드할 수 있습니다.')
 raw=await file.read()
 if len(raw)>10*1024*1024: raise HTTPException(400,'이미지 크기는 10MB 이하여야 합니다.')
 try: Image.open(io.BytesIO(raw)).verify()
 except Exception: raise HTTPException(400,'열 수 없는 이미지 파일입니다.')
 suffix='.png' if file.content_type=='image/png' else '.jpg';report_id=str(uuid.uuid4());filename=f'{report_id}{suffix}'
 (UPLOADS/filename).write_bytes(raw)
 reports=json.loads(REPORTS.read_text(encoding='utf-8')) if REPORTS.exists() else []
 species_id=next((key for key,value in SPECIES.items() if value.get('name')==species_type),'')
 reports.append({'id':report_id,'location':location,'latitude':latitude,'longitude':longitude,'discoveredAt':discovered_at,'speciesId':species_id,'speciesType':species_type,'content':content,'imagePath':f'uploads/{filename}','approved':False,'annotations':[]})
 REPORTS.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
 return {'id':report_id,'message':'제보가 접수되었습니다. 검수 후 지도와 학습 데이터에 반영됩니다.'}
