#!/usr/bin/env python3
"""Audit saved states and compare the same microstructure at two grid spacings."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from evosel3d import ipf_z
from evosel3d.analysis import audit, observables


def section_islands(state):
    owner=state['owner']; j=owner.shape[1]//2
    section=owner[:,j,:]
    count=0; largest=None; largest_size=0
    for grain in np.unique(section[section>=0]):
        components,n=label(section==grain)
        for component in range(1,n+1):
            mask=components==component
            size=int(mask.sum())
            if size<4 or mask[0].any():
                continue
            count+=1
            if size>largest_size and not mask[:,0].any() and not mask[:,-1].any():
                largest_size=size
                cells=np.argwhere(mask)
                k,i=cells[np.argmax(cells[:,0])]
                largest=(int(grain),int(k),int(j),int(i))
    return count,largest


def path_figure(state, point, destination):
    if point is None:
        return None
    g,k,j,i=point
    owner=state['owner']; nz,ny,nx=owner.shape
    p=(k*ny+j)*nx+i
    parents=state['parent'].ravel(); initial=state['initial_owner'].ravel()
    path=[]
    while p>=0:
        kk,rem=divmod(p,ny*nx); jj,ii=divmod(rem,nx)
        path.append([ii,jj,kk])
        if initial[p]>=0:
            break
        p=int(parents[p])
    dx=float(state['dx'])
    path=(np.array(path,dtype=float)+.5)*dx
    # Unwrap periodic axes to avoid spurious long line segments across the box.
    width=nx*dx
    path[:,:2]=np.unwrap(path[:,:2],period=width,axis=0)
    yy=(j+.5)*dx
    rgb=ipf_z(state['rotations'])
    section=owner[:,j,:]
    image=np.ones((*section.shape,3))
    mask=section>=0; image[mask]=rgb[section[mask]]
    fig=plt.figure(figsize=(11,5))
    ax=fig.add_subplot(121)
    ax.imshow(image,origin='lower',extent=(0,width,0,nz*dx),interpolation='nearest',aspect='equal')
    ax.scatter((i+.5)*dx,(k+.5)*dx,c='black',s=28)
    ax.set(xlabel='x',ylabel='z',title=f'XZ at y={yy:.3f}; marked grain {g}')
    ax.set_ylim(0,float(state['height'].max())+.5)
    ax3=fig.add_subplot(122,projection='3d')
    ax3.plot(path[:,0],path[:,1],path[:,2],color='black',lw=1.5)
    ax3.scatter(*path[0],c='red',s=40,label='Point in section')
    ax3.scatter(*path[-1],c='blue',s=40,label='Initial seed material')
    xx,zz=np.meshgrid(np.linspace(path[:,0].min()-.2,path[:,0].max()+.2,2),
                       np.linspace(0,path[:,2].max(),2))
    ax3.plot_surface(xx,np.full_like(xx,yy),zz,alpha=.15,color='grey')
    ax3.set(xlabel='x (unwrapped)',ylabel='y (unwrapped)',zlabel='z',
            title='Same-grain causal path leaves the section')
    ax3.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(destination,dpi=170); plt.close(fig)
    return dict(grain_id=g,voxel_count=len(path),section_y=yy,
                path_y_min=float(path[:,1].min()),path_y_max=float(path[:,1].max()),
                initial_seed_z=float(path[-1,2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/3d'))
    args=p.parse_args()
    report={}
    for prefix in ('111','100'):
        states={n:dict(np.load(args.root/f'n{n}'/f'{prefix}_state.npz')) for n in (64,96)}
        np.testing.assert_array_equal(states[64]['centers'],states[96]['centers'])
        np.testing.assert_array_equal(states[64]['rotations'],states[96]['rotations'])
        report[prefix]={}
        for n,state in states.items():
            checks=audit(state)
            if not checks['all_grown_voxels_connected_to_initial_seed']:
                raise AssertionError(checks)
            metrics=observables(state)
            islands,point=section_islands(state)
            report[prefix][str(n)]=dict(audit=checks,
                metrics={k:v for k,v in metrics.items() if k!='fully_filled_horizontal_sections'},
                section_components_not_touching_bottom_min4px=islands)
            if n==96:
                report[prefix]['section_path']=path_figure(state,point,args.root/f'{prefix}_section_connection.png')
        coarse=report[prefix]['64']['metrics']; fine=report[prefix]['96']['metrics']
        report[prefix]['grid_relative_changes']={k:(fine[k]-coarse[k])/coarse[k]
                 for k in ('mean_height','rms_roughness','distinct_surface_grains')}
    # Same initial microstructure in the two kinetic conditions, too.
    a=np.load(args.root/'n96/100_state.npz'); b=np.load(args.root/'n96/111_state.npz')
    np.testing.assert_array_equal(a['centers'],b['centers'])
    np.testing.assert_array_equal(a['rotations'],b['rotations'])
    (args.root/'grid_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,11))
    for row,prefix in enumerate(('100','111')):
        for col,kind in enumerate(('cross_section_xz_ipf','surface_height')):
            axes[row,col].imshow(plt.imread(args.root/'n96'/f'{prefix}_{kind}.png'))
            axes[row,col].axis('off')
    fig.tight_layout(pad=.1);fig.savefig(args.root/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
