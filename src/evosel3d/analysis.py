"""Deterministic audits and observables; use particle IDs, not RGB colours."""
import numpy as np
from .geometry import theta_001


def audit(result):
    owner = result['owner'].ravel()
    initial = result['initial_owner'].ravel()
    parent = result['parent'].ravel()
    arrival = result['arrival'].ravel()
    occupied = owner >= 0
    grown = occupied & (initial < 0)
    indices = np.flatnonzero(grown)
    p = parent[indices]
    valid = (p >= 0) & (p < owner.size)
    wrong_parent = int((~valid).sum())
    pp, ii = p[valid], indices[valid]
    wrong_parent += int((owner[pp] != owner[ii]).sum())
    wrong_time = int((arrival[pp] >= arrival[ii]).sum())
    nz,ny,nx = result['owner'].shape
    dk = np.abs(pp//(ny*nx)-ii//(ny*nx))
    dj = np.abs((pp//nx)%ny-(ii//nx)%ny)
    di = np.abs(pp%nx-ii%nx)
    manhattan = dk + np.minimum(dj,ny-dj) + np.minimum(di,nx-di)
    nonadjacent = int(np.count_nonzero(manhattan != 1))
    # Strictly decreasing time on a face-adjacent same-ID chain in a finite
    # voxel set proves every grown voxel reaches an initial seed voxel.
    seed_changed = int(np.count_nonzero((initial >= 0) & (owner != initial)))
    roots_off_substrate = 0
    split_seeds = 0
    nz,ny,nx = result['owner'].shape
    for g in np.unique(initial[initial>=0]):
        remaining = set(np.flatnonzero(initial==g).tolist())
        components = 0
        while remaining:
            start = remaining.pop()
            stack = [start]
            touches_substrate = False
            components += 1
            while stack:
                point = stack.pop()
                k, rem = divmod(point,ny*nx)
                j,i = divmod(rem,nx)
                touches_substrate |= k==0
                neighbours = [(k,j,(i-1)%nx),(k,j,(i+1)%nx),
                              (k,(j-1)%ny,i),(k,(j+1)%ny,i),
                              (k-1,j,i),(k+1,j,i)]
                for kk,jj,ii0 in neighbours:
                    if not 0<=kk<nz:
                        continue
                    q = (kk*ny+jj)*nx+ii0
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
            roots_off_substrate += int(not touches_substrate)
        split_seeds += int(components!=1)
    return dict(grown_voxels=len(indices), invalid_predecessors=wrong_parent,
                noncausal_predecessors=wrong_time, nonadjacent_predecessors=nonadjacent,
                changed_seed_voxels=seed_changed,
                seed_components_not_touching_substrate=roots_off_substrate,
                disconnected_initial_seeds=split_seeds,
                all_grown_voxels_connected_to_initial_seed=(wrong_parent==0 and wrong_time==0
                    and nonadjacent==0 and seed_changed==0 and roots_off_substrate==0 and split_seeds==0))


def observables(result):
    height = result['height']
    top = result['surface_owner']
    theta = theta_001(result['rotations'])
    valid = top >= 0
    angles = theta[top[valid]]
    normals_z = np.abs(result['rotations'][:,2,:])
    theta111 = np.degrees(np.arccos(np.clip(normals_z.sum(axis=1)/np.sqrt(3),0,1)))
    owner = result['owner']
    dx = float(result['dx'])
    width = owner.shape[2]*dx
    # Paper A.1: horizontal sections below ALL local surface heights.
    sections = []
    for k in range(owner.shape[0]):
        tags = owner[k]
        if np.all(tags >= 0):
            number = len(np.unique(tags))
            sections.append([(k+.5)*dx, number, width/np.sqrt(number)])
    return dict(mean_height=float(height.mean()), rms_roughness=float(height.std()),
                xy_coverage_fraction=float(valid.mean()),
                height_statistics_include_bare_substrate_as_zero=bool(not valid.all()),
                minimum_height=float(height.min()), maximum_height=float(height.max()),
                distinct_surface_grains=int(len(np.unique(top[valid]))),
                projected_area_weighted_theta001_mean_deg=float(angles.mean()),
                projected_area_weighted_theta001_median_deg=float(np.median(angles)),
                surface_fraction_within_15deg_001=float((angles<15).mean()),
                surface_fraction_within_15deg_111=float((theta111[top[valid]]<15).mean()),
                initial_number_fraction_within_15deg_001=float((theta<15).mean()),
                initial_number_fraction_within_15deg_111=float((theta111<15).mean()),
                surface_fraction_theta001_above_45deg=float((angles>45).mean()),
                fully_filled_horizontal_sections=sections)
