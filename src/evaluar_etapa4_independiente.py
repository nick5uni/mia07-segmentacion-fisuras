"""Evaluate frozen team checkpoints; no training or selection on TEST."""
from pathlib import Path
import ast, csv, json, time, hashlib, platform, sys
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/recalculado_independiente'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
ns={'torch':torch,'nn':nn}
source=ROOT/'src/models.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]),str(source),'exec'),ns)
rows=list(csv.DictReader((ROOT/'data/listas/test.csv').open(encoding='utf-8-sig')))
assert len(rows)==111
xs=[]; ys=[]
for r in rows:
 for key,hkey in [('image_path','image_sha256'),('mask_path','mask_sha256')]:
  assert hashlib.sha256((ROOT/'data/reproducido'/r[key]).read_bytes()).hexdigest()==r[hkey]
 xs.append(np.array(Image.open(ROOT/'data/reproducido'/r['image_path']).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255)
 ys.append(np.array(Image.open(ROOT/'data/reproducido'/r['mask_path']))>0)
X=torch.from_numpy(np.stack(xs));Y=np.stack(ys)
jobs=[]
for model, stage in [('A0','NB02_EncoderDecoder'),('A1','NB03_MiniUNet')]:
 jobs.append((model,20261026,ROOT/'weights'/stage/'checkpoint_best.pt'))
 for seed in [20261027,20261028]:
  jobs.append((model,seed,ROOT/'weights/NB04_Semillas'/f'{model}_seed{seed}'/'checkpoint_best.pt'))
allrows=[];summary=[]
for model,seed,path in jobs:
 start=time.perf_counter()
 # Restrict pickle globals to the NumPy RNG data saved in team checkpoints.
 with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]):
  state=torch.load(path,map_location='cpu',weights_only=True)
 assert state['seed']==seed
 net=ns['EncoderDecoder' if model=='A0' else 'MiniUNet']().eval()
 net.load_state_dict(state['model_state_dict'],strict=True)
 preds=[]
 with torch.inference_mode():
  for j in range(0,len(rows),4):preds.append((torch.sigmoid(net(X[j:j+4]))>=.5).numpy()[:,0])
 P=np.concatenate(preds);conf=[]
 for i,r in enumerate(rows):
  tp=int((P[i]&Y[i]).sum());fp=int((P[i]&~Y[i]).sum());fn=int((~P[i]&Y[i]).sum());tn=int((~P[i]&~Y[i]).sum())
  conf.append([tp,fp,fn,tn]);allrows.append({'model':model,'seed':seed,'image_id':r['image_id'],'tp':tp,'fp':fp,'fn':fn,'tn':tn,'dice':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)})
 a=np.array(conf,dtype=np.int64);tp,fp,fn,tn=a.sum(0)
 with torch.inference_mode():
  for _ in range(3):net(X[:4])
  times=[]
  for _ in range(10):
   t=time.perf_counter();net(X[:4]);times.append((time.perf_counter()-t)*1000/4)
 z={'model':model,'seed':seed,'best_epoch':int(state['epoch']),'parameters':sum(p.numel() for p in net.parameters()),'dice':float(2*tp/(2*tp+fp+fn)),'iou':float(tp/(tp+fp+fn)),'precision':float(tp/(tp+fp)),'recall':float(tp/(tp+fn)),'dice_macro':float(np.mean([x['dice'] for x in allrows if x['model']==model and x['seed']==seed])),'forward_cpu_ms_per_image_median':float(np.median(times)),'elapsed_evaluation_s':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_path':str(path.relative_to(ROOT))}
 summary.append(z);np.savez_compressed(OUT/f'predicciones_{model}_seed{seed}.npz',pred=P,counts=a)
 (OUT/'metricas_corridas.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(z),flush=True)
with (OUT/'metricas_por_imagen.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (OUT/'metricas_corridas.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
env={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'cpu':platform.processor(),'torch_threads':torch.get_num_threads(),'device':'CPU','test_images':111,'threshold':.5,'batch':4,'timing':'forward only, 3 warmup batches and 10 repetitions of first batch, median per image; no preprocessing, no GPU','test_manifest_sha256':hashlib.sha256((ROOT/'data/listas/test.csv').read_bytes()).hexdigest()}
(OUT/'entorno.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
