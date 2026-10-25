#!/usr/bin/env python3
"""Run, audit and plot paper-informed 3D growth; save complete numerical evidence."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evosel3d import Config, simulate, ipf_z, theta_001
from evosel3d.analysis import audit, observables


def plots(result, config, out, prefix):
    colors = ipf_z(result['rotations'])
    owner = result['owner']
    dx = float(result['dx'])
    j = owner.shape[1]//2
    section = owner[:,j,:]
    rgb = np.ones((*section.shape,3))
    mask = section>=0
    rgb[mask] = colors[section[mask]]
    fig, ax = plt.subplots(figsize=(6,8))
    ax.imshow(rgb, origin='lower', interpolation='nearest',
              extent=(0,config.width,0,owner.shape[0]*dx), aspect='equal')
    ax.set_ylim(0, float(result['height'].max())+0.6)
    ax.set(xlabel='x', ylabel='z', title=f'X-Z section, IPF-Z, t={config.end_time:g}\n'
           f'alpha={config.alpha:g}, y={(j+.5)*dx:.3f}')
    fig.tight_layout()
    fig.savefig(out/f'{prefix}_cross_section_xz_ipf.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(result['height'], origin='lower', interpolation='nearest',
                    extent=(0,config.width,0,config.width), cmap='viridis')
    fig.colorbar(im, ax=ax, label='Surface height (cell-centre sampling)')
    ax.set(xlabel='x',ylabel='y',title=f'Surface height, t={config.end_time:g}, alpha={config.alpha:g}')
    fig.tight_layout()
    fig.savefig(out/f'{prefix}_surface_height.png', dpi=180)
    plt.close(fig)
    # An explicit colour key avoids inferring grain IDs or facets from hue.
    n=150
    a,b=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    valid=a<=b
    zc=np.stack((a,b,np.ones_like(a)),axis=-1)
    zc/=np.linalg.norm(zc,axis=-1,keepdims=True)
    rotations=np.zeros((n,n,3,3)); rotations[:,:,2,:]=zc
    rgb=ipf_z(rotations); rgb[~valid]=1
    fig,ax=plt.subplots(figsize=(4,4))
    ax.imshow(rgb,origin='lower',extent=(0,1,0,1),interpolation='nearest')
    ax.text(.01,.02,'001',color='black'); ax.text(.01,.94,'011',color='black')
    ax.text(.84,.94,'111',color='white')
    ax.set(xlabel='a/c',ylabel='b/c',title='Cubic IPF-Z key (explicit RGB map)')
    fig.tight_layout(); fig.savefig(out/'ipf_key.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    bins=np.linspace(0,np.degrees(np.arccos(1/np.sqrt(3))),23)
    theta=theta_001(result['rotations'])
    top=result['surface_owner']; good=top>=0
    ax.hist(theta,bins=bins,density=True,histtype='step',label='Initial seeds (number weighted)')
    ax.hist(theta[top[good]],bins=bins,density=True,histtype='step',label='Surface (projected area weighted)')
    ax.set(xlabel='Angle to nearest <001> (degrees)',ylabel='Density',title=f'Texture, alpha={config.alpha:g}')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out/f'{prefix}_texture.png',dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--nx',type=int,default=64)
    p.add_argument('--width',type=float,default=10.)
    p.add_argument('--grains',type=int,default=121)
    p.add_argument('--time',type=float,default=10.)
    p.add_argument('--seed',type=int,default=2)
    p.add_argument('--seed-radius',type=float,default=.24)
    p.add_argument('--cfl',type=float,default=.6)
    p.add_argument('--band-cells',type=int,default=5)
    p.add_argument('--out',type=Path,default=Path('results/3d'))
    p.add_argument('--prefix',default=None)
    args=p.parse_args()
    config=Config(width=args.width,nx=args.nx,grains=args.grains,alpha=args.alpha,
                  end_time=args.time,seed=args.seed,seed_radius=args.seed_radius,
                  cfl=args.cfl,band_cells=args.band_cells)
    prefix=args.prefix or f'alpha{args.alpha:g}'
    args.out.mkdir(parents=True,exist_ok=True)
    result=simulate(config)
    checks=audit(result)
    if not checks['all_grown_voxels_connected_to_initial_seed']:
        raise AssertionError(checks)
    np.savez_compressed(args.out/f'{prefix}_state.npz',**result)
    summary=dict(config=asdict(config),audit=checks,observables=observables(result),
                 solver='blocked first-order upwind kinetic-Wulff support Hamiltonian; not paper Eq7',
                 reference='10.1016/j.actamat.2004.11.013',
                 dx=float(result['dx']),dt=float(result['dt']),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)))
    (args.out/f'{prefix}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(result,config,args.out,prefix)
    print(json.dumps({k:v for k,v in summary['observables'].items() if k!='fully_filled_horizontal_sections'},indent=2))
    print(checks)


if __name__=='__main__':
    main()
