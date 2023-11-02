import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface
import array_mapper as am
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import csdl
from mirror import Mirror
from expansion_op import ac_expand
from mpl_toolkits.mplot3d import proj3d
from VAST.core.vast_solver_unsteady import VASTSolverUnsteady, PostProcessor
from VAST.core.profile_model import gen_profile_output_list, PPSubmodel
from plot import plot_wireframe



def run_htail(nt, dt, ns, nc, alpha, mach=0.2, plot=True, return_L = False):
    file_name = 'LibertyLifter3.stp'
    caddee = cd.CADDEE()
    caddee.system_model = system_model = cd.SystemModel()
    caddee.system_representation = sys_rep = cd.SystemRepresentation()
    caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
    spatial_rep = sys_rep.spatial_representation
    spatial_rep.import_file(file_name=file_name)
    spatial_rep.refit_geometry(file_name=file_name)
    # spatial_rep.plot(plot_types=['mesh'])



    # region components
    def build_component(name, search_names):
        primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
        component = LiftingSurface(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
        sys_rep.add_component(component)
        return component

    wing = build_component('wing', ['WingGeom'])
    htail = build_component('htail', ['HTail'])
    fuse = build_component('fuse', ['FuselageGeom'])

    # endregion

    # region meshes


    # htail mesh:
    num_spanwise_vlm_htail = ns
    num_chordwise_vlm_htail = nc
    htail_leading_edge = htail.project(np.linspace(np.array([112, -27, 32]), np.array([112, 27, 32]), num_spanwise_vlm_htail), direction=np.array([0., 0., -1.]), plot=False)
    htail_trailing_edge = htail.project(np.linspace(np.array([126, -27, 32]), np.array([126, 27, 32]), num_spanwise_vlm_htail), direction=np.array([0., 0., -1.]), plot=False)
    htail_chord_surface = am.linspace(htail_leading_edge, htail_trailing_edge, num_chordwise_vlm_htail)
    # spatial_rep.plot_meshes([htail_chord_surface])
    htail_upper_surface_wireframe = htail.project(htail_chord_surface.value + np.array([0., 0., 2.]), direction=np.array([0., 0., -2.]), grid_search_n=30, plot=False)
    htail_lower_surface_wireframe = htail.project(htail_chord_surface.value - np.array([0., 0., 2.]), direction=np.array([0., 0., 2.]), grid_search_n=30, plot=False)
    htail_camber_surface = am.linspace(htail_upper_surface_wireframe, htail_lower_surface_wireframe, 1).value

    # endregion

    num_nodes = nt


    # design scenario

    overmodel = m3l.Model()

    design_scenario = cd.DesignScenario(name='wig')

    wig_condition = cd.CruiseCondition(name='wig')
    wig_condition.atmosphere_model = cd.SimpleAtmosphereModel()
    wig_condition.set_module_input(name='altitude', val=0)
    wig_condition.set_module_input(name='mach_number', val=mach, dv_flag=True, lower=0.1, upper=0.3)
    wig_condition.set_module_input(name='range', val=1000)
    wig_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0))
    wig_condition.set_module_input(name='flight_path_angle', val=0)
    wig_condition.set_module_input(name='roll_angle', val=0)
    wig_condition.set_module_input(name='yaw_angle', val=0)
    wig_condition.set_module_input(name='wind_angle', val=0)
    wig_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))

    ac_states = wig_condition.evaluate_ac_states()
    ac_expander = ac_expand(num_nodes=nt)
    ac_states_expanded = ac_expander.evaluate(ac_states)

    rotation_point = np.array([0,0,0])

    non_rotor_surfaces = []
    # tail mirroring
    tail_mirror_model = Mirror(component=htail, mesh_name='htail_mesh', nt=nt, ns=num_spanwise_vlm_htail, nc=num_chordwise_vlm_htail, point=rotation_point, mesh=htail_camber_surface[0,:,:,:]*0.3048)
    tail_mesh = tail_mirror_model.evaluate()[0]
    non_rotor_surfaces.append(tail_mesh)

    uvlm_parameters = [('u',True,ac_states_expanded['u']),
                        ('v',True,ac_states_expanded['v']),
                        ('w',True,ac_states_expanded['w']),
                        ('p',True,ac_states_expanded['p']),
                        ('q',True,ac_states_expanded['q']),
                        ('r',True,ac_states_expanded['r']),
                        ('theta',True,ac_states_expanded['theta']),
                        ('psi',True,ac_states_expanded['psi']),
                        ('gamma',True,ac_states_expanded['gamma']),
                        ('psiw',True,np.zeros((nt, 1)))]

    # ode stuff for props
    surface_names = []
    surface_shapes = []
    initial_conditions = []

    # ode stuff and interactions for non-rotors:
    for surface in non_rotor_surfaces:
        # ode parameters and ICs
        surface_names.append(surface.name)
        num_chordwise = surface.shape[1]
        num_spanwise = surface.shape[2]
        surface_shapes.append(surface.shape[1:4])
        uvlm_parameters.append((surface.name, True, surface))
        # uvlm_parameters.append((surface.name+'_coll_vel', True, np.zeros((nt, num_chordwise-1, num_spanwise-1, 3))))
        initial_conditions.append((surface.name+'_gamma_w_0', np.zeros((nt-1, num_spanwise-1))))
        initial_conditions.append((surface.name+'_wake_coords_0', np.zeros((nt-1, num_spanwise, 3))))

    pp_vars = []
    pp_vars.append((non_rotor_surfaces[0].name+'_C_L', (nt,1)))
    pp_vars.append((non_rotor_surfaces[0].name+'_C_D_i', (nt,1)))


    profile_outputs = gen_profile_output_list(surface_names, surface_shapes)
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
    post_processor = PPSubmodel(surface_names = surface_names, ode_surface_shapes=ode_surface_shapes, delta_t=dt, nt=num_nodes+1, symmetry=False)

    model = m3l.DynamicModel()
    uvlm = VASTSolverUnsteady(num_nodes = num_nodes, 
                            surface_names = surface_names, 
                            surface_shapes = surface_shapes, 
                            delta_t = dt, 
                            nt = nt+1,
                            sub = False,
                            free_wake=True,)
    uvlm_residual = uvlm.evaluate()
    model.register_output(uvlm_residual)
    model.set_dynamic_options(initial_conditions=initial_conditions,
                            num_times=num_nodes,
                            h_stepsize=dt,
                            parameters=uvlm_parameters,
                            int_naming=('op_',''),
                            integrator='ForwardEuler',
                            approach='time-marching',
                            copycat_profile=True,
                            profile_outputs=profile_outputs,
                            post_processor=post_processor,
                            pp_vars=pp_vars)
    uvlm_op = model.assemble(return_operation=True)
    C_L, C_D_i = uvlm_op.evaluate()[0:len(pp_vars)]

    overmodel.register_output(C_L)
    overmodel.register_output(C_D_i)


    # add the cruise m3l model to the cruise condition
    wig_condition.add_m3l_model('wig_model', overmodel)
    # # add the design condition to the design scenario
    design_scenario.add_design_condition(wig_condition)
    system_model.add_design_scenario(design_scenario=design_scenario)
    caddee_csdl_model = caddee.assemble_csdl()
    # caddee_csdl_model = overmodel.assemble_csdl()

    model_csdl = caddee_csdl_model

    h = model_csdl.create_input('height_above_water', val=10)
    pitch_angle = model_csdl.create_input('aircraft_pitch', val=np.deg2rad(alpha))

    model_csdl.connect('height_above_water', 'system_model.wig.wig.wig.operation.input_model.htail_meshmirror.h')
    model_csdl.connect('aircraft_pitch', 'system_model.wig.wig.wig.operation.input_model.htail_meshmirror.theta')

    sim = Simulator(model_csdl, analytics=True, lazy=1)
    sim.run()

    C_L_integrated = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.htail_mesh_out_C_L']
    C_D_i_integrated = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.htail_mesh_out_C_D_i']

    L_integrated = sim['system_model.wig.wig.wig.operation.post_processor.ThrustDrag.htail_mesh_out_L']

    # print(np.sum(sim['system_model.wig.wig.wig.operation.prob.htail_mesh_out_s_panel'][0,:,:]))
    
    if plot:
        plot_wireframe(sim, surface_names, nt, plot_mirror=True, interactive=False)

    del sim
    if return_L:
        return C_L_integrated, C_D_i_integrated, L_integrated
    else:
        return C_L_integrated, C_D_i_integrated


import pickle
# sweeps
sweep_nt = False
sweep_nc = False
sweep_nc_coarse = False
sweep_ns = False
sweep_aoa = False

# plots
plot_nt = True
plot_time = False
plot_nc = False
plot_nc_coarse = False
plot_ns = False
plot_aoa = False


nt0 = 25
nc0 = 25
ns0 = 15

nt_list = [5,10,20,30,40,50,75,100]
nc_list = [2,5,10,20,30,40,50,75,100,125,150,175,200]
nc_coarse_list = [50,75,100,125,150,175,200]
ns_list = [2,5,10,20,30,40,50,75,100]
aoa_list = np.linspace(-15,15,11)

def save_data(CL, CD, fname):
    list = [CL, CD]
    open_file = open(fname, "wb")
    pickle.dump(list, open_file)
    open_file.close()

# nt = nt0
# t_end = 1.
# dt = t_end/nt
# CL, CD = run_htail(nt=nt, dt = dt, ns = ns0, nc = nc0, alpha=5, plot=True)
# print(CL[-1])
# save_data(CL, CD, 'test3.pickle')

# file = open('test2.pickle', 'rb')
# data = pickle.load(file)
# file.close()

# CL_history = data[0]
# CD_history = data[1]
# times = np.linspace(0,1,20)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, sharex=True)
# ax[0].plot(times, CL_history, '-', color='black')
# ax[0].set_ylabel('C_L')
# ax[0].grid()
# ax[0].set_ylim([0.2, 0.4])

# ax[1].plot(times, CD_history, '-', color='black')
# plt.xlabel('time (s)')
# ax[1].set_ylabel('C_D_i')
# ax[1].grid()
# ax[1].set_ylim([0.0, 0.04])
# # ax.set_xlim([0, 100])
# # ax.set_ylim([0, .01])
# plt.show()


if sweep_nt:
    # sweep of nt:
    t_end = 1.
    CL_list = []
    CD_list = []
    for nt in nt_list:
        dt = t_end/nt
        CL, CD = run_htail(nt=nt, dt = dt, ns = ns0, nc = nc0, alpha=5, plot=False)
        CL_list.append(CL)
        CD_list.append(CD)
    save_data(CL_list, CD_list, 'dt_sweep.pickle')

if sweep_nc:
    # sweep of nc:
    t_end = 1.
    nt = nt0
    CL_list = []
    CD_list = []
    L_list = []
    for nc in nc_list:
        dt = t_end/nt
        CL, CD, L = run_htail(nt=nt, dt = dt, ns = ns0, nc = nc, alpha=5, plot=False, return_L = True)
        CL_list.append(CL)
        CD_list.append(CD)
        L_list.append(L)
    save_data(CL_list, CD_list, 'nc_sweep_vc.pickle')

if sweep_nc_coarse:
    # sweep of nc:
    t_end = 1.
    nt = 10
    CL_list = []
    CD_list = []
    L_list = []
    for nc in nc_coarse_list:
        dt = t_end/nt
        CL, CD, L = run_htail(nt=nt, dt = dt, ns = 5, nc = nc, alpha=5, plot=False, return_L = True)
        CL_list.append(CL)
        CD_list.append(CD)
        L_list.append(L)
    save_data(CL_list, CD_list, 'nc_coarse_sweep2.pickle')

if sweep_ns:
    # sweep of ns:
    t_end = 1.
    nt = nt0
    nc = nc0
    CL_list = []
    CD_list = []
    for ns in ns_list:
        dt = t_end/nt
        CL, CD = run_htail(nt=nt, dt = dt, ns = ns, nc = nc, alpha=5, plot=False)
        CL_list.append(CL)
        CD_list.append(CD)

    save_data(CL_list, CD_list, 'ns_sweep.pickle')

if sweep_aoa:
    # sweep of aoa:
    t_end = 1.
    nt = nt0
    nc = nc0
    ns = ns0
    CL_list = []
    CD_list = []
    for alpha in aoa_list:
        dt = t_end/nt
        CL, CD = run_htail(nt=nt, dt = dt, ns = ns, nc = nc, alpha=alpha, plot=False)
        CL_list.append(CL)
        CD_list.append(CD)

    save_data(CL_list, CD_list, 'aoa_sweep.pickle')

def plot(fname, dname, x, log):
    file = open(fname, 'rb')
    data = pickle.load(file)
    file.close()

    CL_list = data[0]
    CD_list = data[1]

    CL_tf = [cl[-1] for cl in CL_list]
    CD_tf = [cd[-1] for cd in CD_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(x, CL_tf, 'o-', color='black')
    ax[0].set_ylabel('C_L')
    ax[0].grid()

    ax[1].plot(x, CD_tf, 'o-', color='black')
    plt.xlabel(dname)
    ax[1].set_ylabel('C_D_i')
    ax[1].grid()
    if log:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
    # ax.set_xlim([0, 100])
    # ax.set_ylim([0, .01])
    plt.show()

    return CL_tf, CD_tf

if plot_nt:
    plot('dt_sweep.pickle', 'nt', nt_list, False)

if plot_time:
    file = open('dt_sweep.pickle', 'rb')
    data = pickle.load(file)
    file.close()

    CL_history = data[0][-1]
    CD_history = data[1][-1]
    times = np.linspace(0,1,nt_list[-1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(times, CL_history, '-', color='black')
    ax[0].set_ylabel('C_L')
    ax[0].grid()
    ax[0].set_ylim([0.2, 0.4])

    ax[1].plot(times, CD_history, '-', color='black')
    plt.xlabel('time (s)')
    ax[1].set_ylabel('C_D_i')
    ax[1].grid()
    ax[1].set_ylim([0.0, 0.04])
    # ax.set_xlim([0, 100])
    # ax.set_ylim([0, .01])
    plt.show()


if plot_nc:
    plot('nc_sweep_vc.pickle', 'nc', nc_list, False)

if plot_nc_coarse:
    plot('nc_coarse_sweep2.pickle', 'nc', nc_coarse_list, False)

if plot_ns:
    plot('ns_sweep.pickle', 'ns', ns_list, False)

if plot_aoa:
    CL_tf, CD_tf = plot('aoa_sweep.pickle', 'aoa', aoa_list, False)

    cl_coeffs = np.polyfit(aoa_list, CL_tf, 1)
    print(cl_coeffs)
    # 0.05251592, 0.00018381

    cd_coeffs = np.polyfit(aoa_list, CD_tf, 2)
    print(cd_coeffs)
    # 0.00021278, 0.00000117, 0.00015929

