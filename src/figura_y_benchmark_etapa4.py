import ast,json,time
from pathlib import Path
import torch,numpy as np
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1];O=R/'results/recalculado_independiente';F=O/'figuras';F.mkdir(exist_ok=True)
ns={'torch':torch,'nn':nn};tree=ast.parse((R/'src/models.py').read_text(encoding='utf-8'));exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]),'models','exec'),ns)
torch.set_num_threads(4);torch.manual_seed(42);x=torch.rand(4,3,416,416)
nets={m:ns[c]().eval() for m,c in [('A0','EncoderDecoder'),('A1','MiniUNet')]};times={m:[] for m in nets}
with torch.inference_mode():
 for _ in range(10):
  for n in nets.values():n(x)
 for rep in range(20):
  for m in (['A0','A1'] if rep%2==0 else ['A1','A0']):
   t=time.perf_counter();nets[m](x);times[m].append((time.perf_counter()-t)*1000/4)
(O/'benchmark_cpu.json').write_text(json.dumps({**{m:float(np.median(t)) for m,t in times.items()},'raw_ms_per_image':times,'weights':'random fixed seed for shape-only forward cost; same operations as trained models','warmup':10,'repetitions':20,'batch':4,'threads':4,'scope':'CPU forward only, alternating order'},indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(6.5,5.0));ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
def box(x,y,text):
 ax.add_patch(FancyBboxPatch((x-.95,y-.45),1.9,.9,boxstyle='round,pad=.05',facecolor='#e8eef2',edgecolor='#334f61'));ax.text(x,y,text,ha='center',va='center',fontsize=10)
def arrow(a,b,color='#334f61',style='-'):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.6,'linestyle':style})
for x,y,t in [(1.5,6.8,'Entrada RGB\n416×416×3'),(1.5,5.3,'Encoder 1\n416×416×16'),(1.5,3.6,'Encoder 2\n208×208×32'),(5,1.7,'Parte central\n104×104×64'),(8.5,3.6,'Decoder 1\n208×208×32'),(8.5,5.3,'Decoder 2\n416×416×16'),(8.5,6.8,'Logits / máscara\n416×416×1')]:box(x,y,t)
for a,b in [((1.5,6.3),(1.5,5.8)),((1.5,4.8),(1.5,4.1)),((1.5,3.1),(4,1.7)),((6,1.7),(8.5,3.1)),((8.5,4.1),(8.5,4.8)),((8.5,5.8),(8.5,6.3))]:arrow(a,b)
for y in [3.6,5.3]:arrow((2.5,y),(7.5,y),'#b85828','--')
ax.text(5,7.6,'A0: camino sólido · A1: añade caminos discontinuos',ha='center',fontsize=10);ax.text(5,4.35,'Concatenaciones skip\nsolo en A1',ha='center',color='#9c451f',fontsize=10)
fig.tight_layout();fig.savefig(F/'arquitectura.png',dpi=180,bbox_inches='tight')
