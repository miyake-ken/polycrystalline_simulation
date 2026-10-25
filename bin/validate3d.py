#!/usr/bin/env python3
"""Numerical validation of isolated Wulff growth and temporal/grid sensitivity.

Run independently of plotting. No visual fitting or changes to seed distributions.
The fixed physical finite seed and orientation are shared across resolutions.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from evosel3d import Config, facets, simulate, wulff_vertices
from evosel3d.analysis import audit, observables
from evosel3d.geometry import initial_nuclei


def isolated(nx, alpha, cfl=.6, band=5, end_time=1.5):
    cfg=Config(width=6.,nx=nx,grains=1,alpha=alpha,end_time=end_time,
               seed_radius=.5,initial_shape='wulff',cfl=cfl,band_cells=band)
    centers=np.array([[3.,3.,0.]])
    rotations=Rotation.from_rotvec([[.21,-.32,.17]]).as_matrix()
    result=simulate(cfg,centers,rotations,progress=False)
    dx=float(result['dx']); shape=result['owner'].shape
    normals,speeds=facets(alpha)
    verts=wulff_vertices(alpha)
    seedtime=cfg.seed_radius/np.linalg.norm(verts,axis=1).max()
    z,y,x=np.indices(shape,dtype=float)
    x=(x+.5)*dx-3.; y=(y+.5)*dx-3.; z=(z+.5)*dx
    gauge=np.full(shape,-np.inf)
    for n,w in zip(normals@rotations[0].T,speeds):
        gauge=np.maximum(gauge,(n[0]*x+n[1]*y+n[2]*z)/w)
    exact=gauge<=cfg.end_time+seedtime
    computed=result['owner']>=0
    # Relative symmetric difference in physical volume; not agreement over an
    # overwhelmingly empty computational box.
    difference=np.count_nonzero(exact!=computed)/np.count_nonzero(exact)
    analytic_volume=np.count_nonzero(exact)*dx**3
    measured_volume=np.count_nonzero(computed)*dx**3
    coords=np.column_stack((x[computed],y[computed],z[computed]))
    labnormals=normals@rotations[0].T
    supports=(coords@labnormals.T).max(axis=0)
    valid=labnormals[:,2]>.2
    facet_support=[dict(index=int(f),family='001' if f<6 else '111',
                        measured_support=float(supports[f]),
                        expected_support=float(speeds[f]*(cfg.end_time+seedtime)),
                        facet_speed=float(speeds[f])) for f in np.flatnonzero(valid)]
    return dict(nx=nx,alpha=alpha,cfl=cfl,band_cells=band,dx=dx,end_time=end_time,
                upper_facet_support=facet_support,
                relative_symmetric_difference=float(difference),
                analytic_voxel_volume=float(analytic_volume),computed_volume=float(measured_volume),
                topology=audit(result)),result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/3d/validation.json'))
    p.add_argument('--resolutions',type=int,nargs='+',default=[32,48,64])
    args=p.parse_args()
    report={'method':'blocked support-Hamiltonian upwind approximation, not exact Smereka Eq7',
            'isolated':[],'time_sensitivity':[],'band_sensitivity':[], 'facet_rate_checks':[]}
    for alpha in (1.1,2.9):
        for nx in args.resolutions:
            data,state=isolated(nx,alpha)
            report['isolated'].append(data)
            print(json.dumps({k:v for k,v in data.items() if k!='topology'}),flush=True)
        for nx in (args.resolutions[0],args.resolutions[-1]):
            early,_=isolated(nx,alpha,end_time=.75)
            late=next(d for d in report['isolated'] if d['nx']==nx and d['alpha']==alpha)
            for e,l in zip(early['upper_facet_support'],late['upper_facet_support']):
                rate=(l['measured_support']-e['measured_support'])/.75
                report['facet_rate_checks'].append(dict(alpha=alpha,nx=nx,family=e['family'],
                    facet_index=e['index'],expected_speed=e['facet_speed'],measured_speed=rate,
                    relative_error=(rate-e['facet_speed'])/e['facet_speed']))
        nx=args.resolutions[-1]
        baseline,base=isolated(nx,alpha)
        half,new=isolated(nx,alpha,cfl=.3)
        union=(base['owner']>=0)|(new['owner']>=0)
        report['time_sensitivity'].append(dict(alpha=alpha,nx=nx,
            changed_occupied_fraction=float(np.count_nonzero(base['owner']!=new['owner'])/union.sum()),
            cfl06_error=baseline['relative_symmetric_difference'],cfl03_error=half['relative_symmetric_difference']))
        wider,new=isolated(nx,alpha,band=8)
        union=(base['owner']>=0)|(new['owner']>=0)
        report['band_sensitivity'].append(dict(alpha=alpha,nx=nx,
            changed_occupied_fraction=float(np.count_nonzero(base['owner']!=new['owner'])/union.sum()),
            band5_error=baseline['relative_symmetric_difference'],band8_error=wider['relative_symmetric_difference']))
    report['grid_error_decreases']={str(a):all(y<x for x,y in zip(
        [r['relative_symmetric_difference'] for r in report['isolated'] if r['alpha']==a][:-1],
        [r['relative_symmetric_difference'] for r in report['isolated'] if r['alpha']==a][1:])) for a in (1.1,2.9)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('isolated',)},indent=2))


if __name__=='__main__':
    main()
