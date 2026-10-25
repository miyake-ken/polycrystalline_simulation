"""Independent geometry, transport, collision and topology checks for the 3D solver."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

from evosel3d import Config, facets, ipf_z, simulate, wulff_vertices
from evosel3d.geometry import initial_nuclei, theta_001
from evosel3d.solver import advance
from evosel3d.analysis import audit


def test_wulff_limits_and_prescribed_active_facets():
    assert len(wulff_vertices(1)) == 8
    assert len(wulff_vertices(3)) == 6
    assert len(wulff_vertices(1.5)) == 12
    for alpha in (1.1,1.5,2.,2.9):
        n,w=facets(alpha)
        verts=wulff_vertices(alpha)
        assert np.all(n@verts.T <= w[:,None]+1e-10)
        np.testing.assert_allclose((n@verts.T).max(axis=1),w,atol=1e-10)


def test_farthest_vertex_predicts_fiber_axis():
    v1=wulff_vertices(1)[0]
    v3=wulff_vertices(3)[0]
    np.testing.assert_allclose(np.abs(v1)/np.linalg.norm(v1),np.ones(3)/np.sqrt(3))
    assert np.count_nonzero(v3) == 1
    for alpha in (1.1,2.9):
        vertices=wulff_vertices(alpha)
        angle=np.degrees(np.arccos(np.abs(vertices).max(axis=1)/np.linalg.norm(vertices,axis=1)))
        if alpha<1.5:
            expected=np.degrees(np.arctan(np.sqrt(5*alpha**2-12*alpha+9)/alpha))
        else:
            expected=np.degrees(np.arctan((3-alpha)/alpha))
        np.testing.assert_allclose(angle,expected,atol=1e-9)


def test_ipf_endpoints_symmetry_and_orientation_convention():
    # IPF-Z only uses row 2 of crystal-to-specimen R.
    matrices=np.tile(np.eye(3),(3,1,1))
    matrices[:,2,:]=[[0,0,1],[0,1/np.sqrt(2),1/np.sqrt(2)],np.ones(3)/np.sqrt(3)]
    np.testing.assert_allclose(ipf_z(matrices),np.eye(3),atol=1e-7)
    np.testing.assert_allclose(ipf_z(matrices[:,:,::-1]),ipf_z(matrices),atol=1e-7)
    np.testing.assert_allclose(theta_001(matrices),[0,45,54.7356103],atol=1e-6)


def test_orientations_uniform_and_seeds_reproducible():
    # Check SO(3), not a uniform polar angle (which biases the texture).
    _,r=initial_nuclei(10,2000,21,0)
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-12)
    assert np.max(abs(np.mean(r[:,2,:]**2,axis=0)-1/3)) < .025
    c1,r1=initial_nuclei(4,9,2,.6)
    c2,r2=initial_nuclei(4,9,2,.6)
    np.testing.assert_array_equal(c1,c2); np.testing.assert_array_equal(r1,r2)


def test_planar_level_set_update_has_exact_support_speed():
    shape=(9,9,9); dx=.1; dt=.01; band=20.
    z,y,x=np.indices(shape)*dx
    gradient=np.array([.3,-.4,.5])
    field=(gradient[0]*x+gradient[1]*y+gradient[2]*z+2.).astype(np.float32)
    phi=field[None].copy()
    owner=np.full(shape,-1,np.int32)
    verts=wulff_vertices(1.1)[None]
    boxes=np.array([[0,9,0,9,0,9]],dtype=np.int64)
    arrival=np.full(shape,np.inf); parent=np.full(shape,-1,np.int64)
    advance(phi,owner,verts,boxes,np.ones(shape,bool),dt,dx,band,0.,arrival,parent)
    speed=np.max(verts[0]@gradient)
    np.testing.assert_allclose(phi[0,2:-2,2:-2,2:-2],
                               field[2:-2,2:-2,2:-2]-dt*speed,atol=3e-7)


def _barrier_run(blocked):
    shape=(18,5,5); dx=.1; dt=.02; band=1.
    z=np.indices(shape)[0]*dx
    phi=np.clip(z-.25,-band,band).astype(np.float32)[None]
    owner=np.where(z<.25,0,-1).astype(np.int32)
    if blocked:
        owner[6]=1
    vertices=np.array(list(product((-1.,1.),repeat=3)))[None]
    boxes=np.array([[0,18,0,5,0,5]],np.int64)
    arrival=np.where(owner>=0,0.,np.inf); parent=np.full(shape,-1,np.int64)
    for step in range(45):
        owner,boxes=advance(phi,owner,vertices,boxes,np.ones(shape,bool),dt,dx,band,
                            step*dt,arrival,parent)
    return owner


def test_buried_front_cannot_reappear_beyond_impenetrable_layer():
    blocked=_barrier_run(True); free=_barrier_run(False)
    assert np.all(blocked[6]==1)
    assert not np.any(blocked[7:]==0)
    assert np.any(free[7:]==0)


def test_simulation_causal_material_paths_and_periodic_translation():
    cfg=Config(width=3,nx=24,grains=3,seed_radius=.3,end_time=.6)
    c,r=initial_nuclei(3,3,2,.63)
    first=simulate(cfg,c,r,progress=False)
    checks=audit(first)
    assert checks['grown_voxels']>0
    assert checks['all_grown_voxels_connected_to_initial_seed']
    shift=5
    c[:,0]=(c[:,0]+shift*cfg.width/cfg.nx)%cfg.width
    second=simulate(cfg,c,r,progress=False)
    np.testing.assert_array_equal(np.roll(first['owner'],shift,axis=2),second['owner'])
    np.testing.assert_allclose(np.roll(first['arrival'],shift,axis=2),second['arrival'],atol=1e-12)


def test_seed_ids_never_silently_disappear():
    cfg=Config(width=4,nx=32,grains=9,seed_radius=.3,end_time=.4)
    result=simulate(cfg,progress=False)
    assert len(np.unique(result['initial_owner'][result['initial_owner']>=0]))==9
    assert audit(result)['changed_seed_voxels']==0


def test_valid_3d_material_can_have_a_disconnected_2d_section():
    from scipy.ndimage import label
    shape=(4,3,3)
    owner=np.full(shape,-1,np.int32)
    initial=owner.copy()
    parent=np.full(shape,-1,np.int64)
    arrival=np.full(shape,np.inf)
    path=[(0,1,1),(1,1,1),(1,2,1),(2,2,1),(3,2,1),(3,1,1)]
    for step,voxel in enumerate(path):
        owner[voxel]=0
        arrival[voxel]=step
        if step:
            parent[voxel]=np.ravel_multi_index(path[step-1],shape)
        else:
            initial[voxel]=0
    state=dict(owner=owner,initial_owner=initial,parent=parent,arrival=arrival)
    assert label(owner[:,1,:]>=0)[1]==2
    assert audit(state)['all_grown_voxels_connected_to_initial_seed']
    parent[path[-1]]=np.ravel_multi_index(path[0],shape)
    assert not audit(state)['all_grown_voxels_connected_to_initial_seed']
    assert audit(state)['nonadjacent_predecessors']==1
    parent[path[-1]]=-1
    assert not audit(state)['all_grown_voxels_connected_to_initial_seed']


def test_crossing_arbitration_uses_full_precision_and_deterministic_ties():
    from evosel3d.solver import _resolve
    owner=np.full((1,3,5),-1,np.int32)
    owner[0,1,1]=0; owner[0,1,3]=1
    boxes=np.array([[0,1,0,3,0,5],[0,1,0,3,0,5]],np.int64)
    candidates=np.full((2,1,3,5),2.,np.float64)
    candidates[:,0,1,2]=[.5+1e-10,.5]
    arrival=np.full(owner.shape,np.inf); parent=np.full(owner.shape,-1,np.int64)
    result=_resolve(owner,boxes,candidates,0.,1.,arrival,parent)
    assert result[0,1,2]==1
    candidates[:,0,1,2]=.5
    result=_resolve(owner,boxes,candidates,0.,1.,arrival,parent)
    assert result[0,1,2]==0
