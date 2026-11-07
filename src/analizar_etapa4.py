from pathlib import Path
import json,csv,platform
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'results/recalculado_independiente';F=O/'figuras';F.mkdir(exist_ok=True)
S=json.loads((O/'metricas_corridas.json').read_text());seeds=[20261026,20261027,20261028]
counts=np.stack([np.stack([np.load(O/f'predicciones_{m}_seed{s}.npz')['counts'] for s in seeds]) for m in ['A0','A1']])
def dice(c):return 2*c[...,0]/np.maximum(2*c[...,0]+c[...,1]+c[...,2],1)
stats={}
for m in ['A0','A1']:
 a=[x for x in S if x['model']==m];stats[m]={k:{'mean':float(np.mean([x[k] for x in a])),'sd':float(np.std([x[k] for x in a],ddof=1))} for k in ['dice','iou','precision','recall','dice_macro','forward_cpu_ms_per_image_median']}
delta=np.array([x['dice'] for x in S if x['model']=='A1'])-np.array([x['dice'] for x in S if x['model']=='A0'])
di=dice(counts);diff=di[1].mean(0)-di[0].mean(0)
rng=np.random.default_rng(20261103);boots=[]
for _ in range(3000):
 ix=rng.integers(0,111,111);ss=rng.integers(0,3,3)
 c=counts[:,ss][:,:,ix].sum(2);boots.append(float((dice(c)[1]-dice(c)[0]).mean()))
st={'models':stats,'paired_seed_delta_dice':delta.tolist(),'mean_delta_dice':float(delta.mean()),'bootstrap_hierarchical95':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_replicates':3000,'bootstrap_seed':20261103,'wilcoxon_mean_image_dice_exploratory':float(wilcoxon(diff,method='approx').pvalue),'images_a1_better':int((diff>0).sum()),'images_a0_better':int((diff<0).sum()),'notes':'Exploratory paired image test, not independent training replications; bootstrap resamples paired seed indices and paired images; apparent source/scene dependence unmodeled; only 3 seeds.'}
rows=list(csv.DictReader((R/'data/listas/test.csv').open(encoding='utf-8-sig')))
fr=np.array([float(x['positive_fraction']) for x in rows]);q=np.quantile(fr,[1/3,2/3]);st['coverage_tercile_bounds']=q.tolist();groups=[]
for j,mask in enumerate([fr<=q[0],(fr>q[0])&(fr<=q[1]),fr>q[1]]):
 groups.append({'group':['Menor área anotada','Área intermedia','Mayor área anotada'][j],'n':int(mask.sum()),'A0_dice_macro':float(di[0,:,mask].mean()),'A1_dice_macro':float(di[1,:,mask].mean())})
st['error_groups']=groups
(O/'estadistica.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
plt.rcParams.update({'font.size':10,'figure.dpi':150})
fig,ax=plt.subplots(figsize=(6.5,3.3))
for m,c in [('A0','#245777'),('A1','#b85828')]:
 a=[x for x in S if x['model']==m];ax.plot([str(s)[-2:] for s in seeds],[x['dice'] for x in a],'o-',label=m,color=c)
ax.set(xlabel='Últimos dos dígitos de la semilla 202610xx',ylabel='Dice global en TEST',title='Resultados de las tres corridas por modelo');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'semillas_test.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,3.5));ax.scatter(di[0].mean(0),di[1].mean(0),s=13,alpha=.7);ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Dice por imagen A0 promediado en 3 semillas',ylabel='Dice por imagen A1 promediado en 3 semillas');ax.set_aspect('equal');fig.tight_layout();fig.savefig(F/'dispersion_test.png');plt.close(fig)
order=np.argsort((di[0,0]+di[1,0])/2);sel=[int(order[0]),int(order[len(order)//2]),int(order[-1])];st['visual_cases']=[{'id':rows[i]['image_id'],'A0':float(di[0,0,i]),'A1':float(di[1,0,i])} for i in sel]
for n,i in enumerate(sel):
 im=np.array(Image.open(R/'data/reproducido'/rows[i]['image_path']).convert('RGB'));y=np.array(Image.open(R/'data/reproducido'/rows[i]['mask_path']))>0
 fig,axs=plt.subplots(1,4,figsize=(10,3))
 axs[0].imshow(im);axs[0].set_title('Fotografía')
 axs[1].imshow(y,cmap='gray');axs[1].set_title('Referencia')
 for j,m in enumerate(['A0','A1']):
  p=np.load(O/f'predicciones_{m}_seed20261026.npz')['pred'][i]
  color=np.zeros_like(im);color[p&y]=[0,190,0];color[p&~y]=[255,40,40];color[~p&y]=[30,100,255]
  overlay=im.copy();mask=p|y;overlay[mask]=(.35*im[mask]+.65*color[mask]).astype('uint8');axs[j+2].imshow(overlay);axs[j+2].set_title(f'{m} Dice {di[j,0,i]:.3f}')
 for ax in axs:ax.axis('off')
 fig.tight_layout();fig.savefig(F/f'caso_{n+1}.png',bbox_inches='tight');plt.close(fig)
(O/'estadistica.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
print(json.dumps(st,indent=2))
