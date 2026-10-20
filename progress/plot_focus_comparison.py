#!/usr/bin/env python3
"""Plot raw scalar histories against the downloaded originals, without smoothing.

Usage:
  python3 tools/plot_focus_comparison.py --run /path/to/run 'Название расчёта'
  python3 tools/plot_focus_comparison.py --auto

Each run must have res/{Imax,Nemax,parameters}.dat. Paths can point directly
at res/. --auto includes only f20_rk2ip and f100_rk2ip production runs.
The available original f20 release covers only 18--19.31 cm; its panel is
explicitly labeled as partial. Missing or empty pending-run files are skipped.
"""
import argparse
from pathlib import Path
import os,re,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/uppe-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--root',type=Path,default=Path('/workspace/results'))
parser.add_argument('--run',nargs=2,action='append',default=[],metavar=('PATH','LABEL'))
parser.add_argument('--auto',action='store_true')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
root=args.root; out=args.output or root/'figures';out.mkdir(parents=True,exist_ok=True)


def read_series(path):
    try:
        if not path.is_file() or path.stat().st_size==0:return np.empty((0,2))
        a=np.loadtxt(path,ndmin=2,comments='#')
    except ValueError:
        # A live process can have an incomplete final write. Keep complete rows.
        rows=[]
        for line in path.read_text().splitlines():
            try:r=[float(x) for x in line.split()]
            except ValueError:continue
            if len(r)>=2 and (not rows or len(r)==len(rows[0])):rows.append(r)
        a=np.asarray(rows)
    if a.ndim!=2 or len(a)==0:return np.empty((0,2))
    return a[np.all(np.isfinite(a),axis=1)]


def load(path,label,params=None,original=False):
    path=Path(path);res=path if path.name=='res' else path/'res'
    params=Path(params) if params else res/'parameters.dat'
    if not params.is_file():return None
    text=params.read_text()
    def value(name):
        m=re.search(r'(?m)^\s*'+re.escape(name)+r'\s*=\s*(\S+)',text)
        if not m:raise ValueError(f'{params}: missing {name}')
        return float(m[1])
    a=read_series(res/'Imax.dat');n=read_series(res/'Nemax.dat')
    if len(a)==0:return None
    return dict(label=label,focus=value('Focus'),energy_mJ=value('W0'),I0_TW=value('I0')*.001,
                I=a,Ne=n,E=read_series(res/'Energy.dat'),original=original,path=str(res))

runs=[]
originals=[('sim-z30-old','latest_seg01','Исходный f=20 (часть18–19,31 см)'),
           ('sim-f100-z1-E5-nopre','latest_seg04','Исходный f=100')]
for tag,seg,label in originals:
    base=root/'original'/tag
    v=load(base/seg,label,base/'grids/res/parameters.dat',True)
    if v:runs.append(v)
paths=list(args.run)
if args.auto:
    for name in ['f20_rk2ip','f100_rk2ip']:
        p=root/name
        if (p/'res/Imax.dat').is_file():paths.append((str(p),name))
seen=set()
for path,label in paths:
    key=str(Path(path).resolve())
    if key in seen:continue
    seen.add(key)
    v=load(path,label)
    if v:
        if not v['original']:
            v['label'] += (f" (начальный участок, расчёт идёт, z≤{v['I'][-1,0]:.3g} см)"
                           if (v['focus']==100 and v['I'][-1,0]<10) or (v['focus']==20 and v['I'][-1,0]<19.5)
                           else f" (доступно z≤{v['I'][-1,0]:.3g} см)")
        runs.append(v)

plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':170})
fig,axs=plt.subplots(2,2,figsize=(13,7.8),sharex='col',constrained_layout=True)
for j,focus in enumerate([20,100]):
    selected=[v for v in runs if abs(v['focus']-focus)<1e-9]
    for v in selected:
        a=v['I'];n=v['Ne'];style={'lw':1.25,'alpha':.9}
        if v['original']:style.update(color='#777777',ls='--',lw=1.4)
        line,=axs[0,j].plot(a[:,0],a[:,1]*v['I0_TW'],label=v['label'],**style)
        if len(n):axs[1,j].plot(n[:,0],n[:,1],color=line.get_color(),**{k:x for k,x in style.items() if k!='color'})
    axs[0,j].set_title(f'f={focus} см — сырые данные')
    axs[0,j].set_ylabel(r'$I_{\max}$, ТВт/см$^2$')
    axs[1,j].set_ylabel(r'$N_{e,\max}/N_0$');axs[1,j].set_xlabel('z, см')
    axs[1,j].ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    if selected:axs[0,j].legend(fontsize=8)
fig.suptitle('Сравнение расчётов без сглаживания\nКонцы новых кривых — доступные данные; снятие артефактов и сходимость ещё требуют проверки')
for ext in ['png','pdf']:fig.savefig(out/f'original_vs_current.{ext}')
plt.close(fig)

# Current f100 diagnostics use only recorded samples, with no interpolation/smoothing.
for v in [x for x in runs if not x['original'] and abs(x['focus']-100)<1e-9]:
    diag=read_series(Path(v['path'])/'OpticalDiagnostics.dat')
    if len(diag)==0 or diag.shape[1]<4:continue
    fig,axs=plt.subplots(3,1,figsize=(11,8.5),sharex=True,constrained_layout=True)
    a=v['I'];n=v['Ne']
    axs[0].plot(a[:,0],a[:,1]*v['I0_TW'],'.-',ms=3,label=r'$\max E_{\mathrm{real}}^2$ — вся оптическая полоса')
    axs[0].plot(diag[:,0],diag[:,1]*v['I0_TW'],'.-',ms=3,label=r'$I_{1,\mathrm{envelope}}$ — первая гармоника')
    axs[0].set_ylabel(r'$I$, ТВт/см$^2$');axs[0].legend(fontsize=9)
    if len(n):axs[1].plot(n[:,0],n[:,1],'.-',ms=3)
    axs[1].set_ylabel(r'$N_{e,\max}/N_0$')
    axs[2].plot(diag[:,0],diag[:,3],'.-',ms=3)
    axs[2].set_ylabel(r'$P_3/P_1$');axs[2].set_xlabel('z, см')
    for ax in axs[1:]:ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    status='расчёт идёт, только начальный участок' if a[-1,0]<10 else 'доступный участок расчёта'
    warning=('Сырые данные; область артефактов z≈30–60 см ещё не достигнута' if a[-1,0]<30
             else 'Сырые данные; сходимость требует отдельного сравнения шага и сеток')
    fig.suptitle(f'f=100 см: {status}, z={a[0,0]:.3g}–{a[-1,0]:.3g} см\n'+warning)
    stem='f100_current_diagnostics' if Path(v['path']).parent.name=='f100_rk2ip' else 'f100_current_diagnostics_'+Path(v['path']).parent.name
    for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}')
    plt.close(fig)

# Machine-readable plot provenance: avoids accidental claims about completed runs.
meta=[]
for v in runs:
    meta.append({k:x for k,x in v.items() if k not in {'I','Ne','E'}}|{'I_rows':len(v['I']),'Ne_rows':len(v['Ne']),
      'z_first_cm':float(v['I'][0,0]),'z_last_cm':float(v['I'][-1,0]),'peak_I_TW_cm2':float(np.max(v['I'][:,1])*v['I0_TW']),
      'peak_Ne_over_N0':float(np.max(v['Ne'][:,1])) if len(v['Ne']) else None})
(out/'original_vs_current.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(out/'original_vs_current.png')
for v in meta:print(f"{v['label']}: {v['I_rows']} rows, z={v['z_first_cm']:.5g}..{v['z_last_cm']:.5g} cm")
