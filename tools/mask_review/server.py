import csv,io,json,threading
from datetime import datetime,timezone
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,Response
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/predicted_mask_review_expanded'
MANIFEST=json.loads((OUT/'manifest.json').read_text())
RECORDS={x['study_id']:x for x in MANIFEST['studies']}
REVIEWS=OUT/'manual_reviews.json'
LOCK=threading.Lock()
app=FastAPI(title='Predicted CVC Mask Review')
def read_reviews():return json.loads(REVIEWS.read_text()) if REVIEWS.exists() else {}
@app.get('/')
def home():return FileResponse(Path(__file__).with_name('index.html'))
@app.get('/api/manifest')
def manifest():
 return {**{k:v for k,v in MANIFEST.items() if k!='studies'},'studies':[{k:v for k,v in r.items() if k not in ['image_path','mask_path','geometry']} for r in RECORDS.values()],'reviews':read_reviews()}
@app.get('/asset/{kind}/{sid}')
def asset(kind:str,sid:str):
 if kind not in ['image','mask'] or sid not in RECORDS:raise HTTPException(404)
 return FileResponse(RECORDS[sid][kind+'_path'])
class Review(BaseModel):
 study_id:str
 status:str
 flags:list[str]=[]
 notes:str=''
@app.post('/api/review')
def review(body:Review,request:Request):
 origin=request.headers.get('origin')
 if origin and origin!=str(request.base_url).rstrip('/'):raise HTTPException(403)
 if body.study_id not in RECORDS or body.status not in ['unreviewed','good','partial','wrong','uncertain','correct_empty']:raise HTTPException(400)
 if not set(body.flags)<={'missed_catheter','extra_foreground','broken_line','tip_region_missing','misalignment'}:raise HTTPException(400)
 with LOCK:
  rows=read_reviews();rows[body.study_id]={**body.model_dump(),'checkpoint_sha256':MANIFEST['checkpoint_sha256'],'threshold':MANIFEST['threshold'],'review_status':'user_visual_review_not_clinical_attestation','updated_at':datetime.now(timezone.utc).isoformat()}
  tmp=REVIEWS.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(REVIEWS)
 return {'saved':True}
@app.get('/reviews.csv')
def export():
 rows=read_reviews();s=io.StringIO();writer=csv.DictWriter(s,fieldnames=['study_id','status','flags','notes','checkpoint_sha256','threshold','review_status','updated_at']);writer.writeheader()
 for sid in RECORDS:
  r=rows.get(sid,dict(study_id=sid,status='unreviewed'));r={**r,'flags':';'.join(r.get('flags',[]))};writer.writerow(r)
 return Response(s.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="cvc_mask_reviews.csv"'})
