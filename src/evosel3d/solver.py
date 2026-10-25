"""Blocked, multi-grain, first-order upwind kinetic-Wulff level-set solver.

This is NOT a transcription of Smereka et al. Eq.(7). It uses the convex
support Hamiltonian H_g(p)=max_{v in R_g W} v.p of the same kinetic Wulff body.
It has the prescribed isolated-crystal facet velocities. Equality with the
paper's nonconvex normal-speed extension under impingement is NOT assumed.

Each grain has its own field. Foreign deposited material is an impermeable
positive-field obstacle; deposited IDs never change. A monotone six-neighbour
stencil and CFL <= 1 mean negative values cannot spontaneously nucleate across
an obstacle. Concurrent crossings compete by interpolated crossing time.
There is no evaluation of a global free-space arrival-time argmin.

Fields are clipped away from the interface. Only boxes near each live front
are updated. Grid diffusion rounds edges; convergence must be measured.
"""
from dataclasses import dataclass
import time

import numpy as np
from numba import njit, prange
from scipy.ndimage import maximum_filter

from .geometry import facets, initial_nuclei, wulff_vertices


@dataclass
class Config:
    width: float = 10.0
    nx: int = 64
    grains: int = 121
    alpha: float = 1.1
    v111: float = 1.0
    end_time: float = 10.0
    seed: int = 2
    seed_radius: float = 0.24
    initial_shape: str = 'sphere'
    cfl: float = 0.6
    band_cells: int = 5


@njit(cache=True)
def _sample(field, owner, g, k, j, i, band):
    nz, ny, nx = owner.shape
    # Periodic in the substrate plane. Reflect at z=0 (no substrate growth).
    k = min(max(k, 0), nz-1)
    j %= ny
    i %= nx
    if owner[k, j, i] >= 0 and owner[k, j, i] != g:
        return band
    return field[k, j, i]


@njit(cache=True, parallel=True)
def _advance_fields(phi, owner, vertices, boxes, near_free, dt, dx, band):
    """Independent grain updates; no shared writes across parallel iterations."""
    nz, ny, nx = owner.shape
    candidates = np.full(phi.shape, 2., dtype=np.float64)
    newboxes = np.full_like(boxes, -1)
    for g in prange(len(phi)):
        k0, k1, j0, j1, i0, i1 = boxes[g]
        if k0 < 0:
            continue
        old = phi[g]
        updated = old[k0:k1, j0:j1, i0:i1].copy()
        lo0, lo1, lo2 = nz, ny, nx
        hi0, hi1, hi2 = -1, -1, -1
        for k in range(k0, k1):
            for j in range(j0, j1):
                for i in range(i0, i1):
                    tag = owner[k, j, i]
                    value = old[k, j, i]
                    if tag >= 0 and tag != g:
                        updated[k-k0, j-j0, i-i0] = band
                        continue
                    if not near_free[k, j, i]:
                        if tag == g:
                            updated[k-k0, j-j0, i-i0] = -band
                        continue
                    xm = _sample(old, owner, g, k, j, i-1, band)
                    xp = _sample(old, owner, g, k, j, i+1, band)
                    ym = _sample(old, owner, g, k, j-1, i, band)
                    yp = _sample(old, owner, g, k, j+1, i, band)
                    zm = _sample(old, owner, g, k-1, j, i, band)
                    zp = _sample(old, owner, g, k+1, j, i, band)
                    dm0, dp0 = (value-xm)/dx, (xp-value)/dx
                    dm1, dp1 = (value-ym)/dx, (yp-value)/dx
                    dm2, dp2 = (value-zm)/dx, (zp-value)/dx
                    h = 0.
                    for v in vertices[g]:
                        h0 = v[0]*(dm0 if v[0]>=0 else dp0)
                        h1 = v[1]*(dm1 if v[1]>=0 else dp1)
                        h2 = v[2]*(dm2 if v[2]>=0 else dp2)
                        h = max(h, h0+h1+h2)
                    new = max(-band, value - dt*h)
                    updated[k-k0, j-j0, i-i0] = new
                    if new < band*(1.-1e-6):
                        lo0, hi0 = min(lo0,k), max(hi0,k)
                        lo1, hi1 = min(lo1,j), max(hi1,j)
                        lo2, hi2 = min(lo2,i), max(hi2,i)
                    if tag < 0 and new <= 0 and value > 0:
                        candidates[g,k,j,i] = value/(value-new)
        old[k0:k1,j0:j1,i0:i1] = updated
        if hi0 >= 0:
            newboxes[g,0] = max(0, lo0-1)
            newboxes[g,1] = min(nz, hi0+2)
            # Boxes touching a periodic seam include the opposite edge.
            newboxes[g,2] = 0 if lo1 <= 1 or hi1 >= ny-2 else lo1-1
            newboxes[g,3] = ny if lo1 <= 1 or hi1 >= ny-2 else hi1+2
            newboxes[g,4] = 0 if lo2 <= 1 or hi2 >= nx-2 else lo2-1
            newboxes[g,5] = nx if lo2 <= 1 or hi2 >= nx-2 else hi2+2
    return candidates, newboxes


@njit(cache=True)
def _resolve(owner, boxes, candidates, now, dt, arrival, parent):
    """Deterministic first-crossing arbitration against the pre-step material."""
    nz,ny,nx=owner.shape
    proposals=owner.copy()
    fractions=np.full(owner.shape,2.,dtype=np.float64)
    for g in range(len(boxes)):
        k0,k1,j0,j1,i0,i1=boxes[g]
        if k0<0:
            continue
        for k in range(k0,k1):
            for j in range(j0,j1):
                for i in range(i0,i1):
                    frac=candidates[g,k,j,i]
                    if frac>=fractions[k,j,i]:
                        continue
                    pred=-1
                    for axis in range(3):
                        for sign in (-1,1):
                            kk,jj,ii=k,j,i
                            if axis==0:
                                ii=(i+sign)%nx
                            elif axis==1:
                                jj=(j+sign)%ny
                            else:
                                kk=k+sign
                            if 0<=kk<nz and owner[kk,jj,ii]==g:
                                pred=(kk*ny+jj)*nx+ii
                    if pred>=0:
                        proposals[k,j,i]=g
                        fractions[k,j,i]=frac
                        parent[k,j,i]=pred
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if owner[k,j,i]<0 and proposals[k,j,i]>=0:
                    arrival[k,j,i]=now+dt*fractions[k,j,i]
    return proposals


def advance(phi, owner, vertices, boxes, near_free, dt, dx, band, now, arrival, parent):
    candidates,newboxes=_advance_fields(phi,owner,vertices,boxes,near_free,dt,dx,band)
    return _resolve(owner,boxes,candidates,now,dt,arrival,parent),newboxes


def initialize(config, centers, rotations):
    dx = config.width/config.nx
    verts = wulff_vertices(config.alpha, config.v111)
    maxspeed = np.linalg.norm(verts, axis=1).max()
    nz = int(np.ceil((maxspeed*config.end_time + config.seed_radius)/dx))+8
    normals, speeds = facets(config.alpha, config.v111)
    velocities = np.array([verts @ r.T for r in rotations])
    seed_time = config.seed_radius/maxspeed
    shape = (nz, config.nx, config.nx)
    band = config.band_cells*dx
    phi = np.full((len(centers), *shape), band, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int32)
    boxes = np.zeros((len(centers),6), dtype=np.int64)
    xy = (np.arange(config.nx)+0.5)*dx
    z = (np.arange(nz)+0.5)*dx
    for g, (center, r) in enumerate(zip(centers, rotations)):
        # Periodic images of seeds on a torus, not clamped outside nuclei.
        ddx = (xy-center[0]+config.width/2) % config.width-config.width/2
        ddy = (xy-center[1]+config.width/2) % config.width-config.width/2
        kz = np.flatnonzero(z-center[2] <= config.seed_radius+2*band)
        local_z = z[kz]-center[2]
        if config.initial_shape == 'sphere':
            # Sphere digitization is face-connected (steps towards the nearest
            # centre reduce distance). A tiny rotated polyhedron can leave
            # disconnected corner voxels even though the continuum body is convex.
            level = np.sqrt(ddx[None,None,:]**2 + ddy[None,:,None]**2
                            + local_z[:,None,None]**2) - config.seed_radius
        elif config.initial_shape == 'wulff':
            level = np.full((len(kz), config.nx, config.nx), -np.inf)
            for n, w in zip(normals @ r.T, speeds):
                distance = (n[0]*ddx[None,None,:] + n[1]*ddy[None,:,None]
                            + n[2]*local_z[:,None,None] - w*seed_time)
                level = np.maximum(level, distance)
        else:
            raise ValueError('initial_shape must be sphere or wulff')
        phi[g,kz] = np.clip(level, -band, band)
        initial = phi[g] <= 0
        if not initial.any() or not initial[0].any():
            raise ValueError(f"seed {g} unresolved; increase nx or seed_radius")
        if np.any(owner[initial] >= 0):
            raise ValueError("initial grains overlap; increase seed separation")
        owner[initial] = g
        indices = np.argwhere(phi[g] < band*(1-1e-6))
        lo, hi = indices.min(axis=0), indices.max(axis=0)
        boxes[g] = [max(0,lo[0]-1), min(nz,hi[0]+2), 0, config.nx, 0, config.nx]
    return phi, owner, velocities, boxes, dx


def surface(owner, dx):
    solid = owner >= 0
    present = solid.any(axis=0)
    k = solid.shape[0]-1-np.argmax(solid[::-1], axis=0)
    height = np.where(present, (k+0.5)*dx, 0.)
    yy, xx = np.indices(k.shape)
    tags = np.where(present, owner[k,yy,xx], -1)
    return height, tags


def simulate(config: Config, centers=None, rotations=None, progress=True):
    if not 0 < config.cfl <= 1 or config.band_cells < 3:
        raise ValueError("require 0<CFL<=1 and band_cells>=3")
    if centers is None:
        centers, rotations = initial_nuclei(config.width, config.grains, config.seed,
                                            2*config.seed_radius + 0.03)
    phi, owner, velocities, boxes, dx = initialize(config, centers, rotations)
    dt = config.cfl*dx / np.abs(velocities).sum(axis=2).max()
    arrival = np.where(owner >= 0, 0., np.inf).astype(np.float64)
    parent = np.full(owner.shape, -1, dtype=np.int64)
    initial_owner = owner.copy()
    from .analysis import audit
    seed_check = audit(dict(owner=owner, initial_owner=initial_owner,
                           arrival=arrival, parent=parent))
    if not seed_check['all_grown_voxels_connected_to_initial_seed']:
        raise ValueError(f'initial digitized seeds are disconnected: {seed_check}')
    now, step = 0., 0
    history = []
    clock = time.monotonic()
    while now < config.end_time - 1e-12:
        actual_dt = min(dt, config.end_time-now)
        near = maximum_filter((owner < 0).astype(np.uint8),
                              size=2*config.band_cells+1,
                              mode=('nearest','wrap','wrap')).astype(bool)
        live = np.bincount(owner[(owner >= 0) & near], minlength=len(centers)) > 0
        boxes[~live] = -1
        previous = owner
        owner, boxes = advance(phi, owner, velocities, boxes, near, actual_dt, dx,
                              config.band_cells*dx, now, arrival, parent)
        if np.any((previous >= 0) & (owner != previous)):
            raise AssertionError("a deposited grain ID changed")
        now += actual_dt
        step += 1
        if step % 25 == 0 or now >= config.end_time-1e-12:
            height, top = surface(owner, dx)
            history.append([now, height.mean(), height.std(),
                            len(np.unique(top[top >= 0]))])
            if progress:
                print(f"alpha={config.alpha:g} step={step} t={now:.3f} "
                      f"h={height.mean():.3f} surface_grains={history[-1][-1]} "
                      f"elapsed={time.monotonic()-clock:.1f}s", flush=True)
        if np.any(owner[-2:] >= 0):
            raise RuntimeError("growth reached top boundary; domain must be enlarged")
    height, top = surface(owner, dx)
    return dict(owner=owner, arrival=arrival, parent=parent, initial_owner=initial_owner,
                centers=centers, rotations=rotations, height=height, surface_owner=top,
                history=np.array(history), dx=np.array(dx), dt=np.array(dt))
