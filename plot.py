import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show, Mesh
import numpy as np

def plot_wireframe(sim, surface_names, nt, interactive = False, plot_mirror = True, wake_color='cyan', rotor_wake_color='red', surface_color='gray', cmap='jet', absolute=True, side_view=False, name='sample_gif', backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(0,25),
        yrange=(-30, 30),
        zrange=(0, 10),
    )
    video = Video(name+".gif", fps=10, backend=backend)
    # first get min and max gamma value:
    min_gamma = 1e100
    max_gamma = -1e100
    for i in range(1, nt-1):
        for surface_name in surface_names:
            if 'mirror' in surface_name and not plot_mirror:
                pass
            else:
                gamma_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_gamma_w'][i, 0:i, :], (-1,1))
                if absolute:
                    gamma_w = np.absolute(gamma_w)
                min_gamma = np.min([min_gamma, np.min(gamma_w)])
                max_gamma = np.max([max_gamma, np.max(gamma_w)])
    for i in range(1, nt - 1):
        vp = Plotter(
            bg='beige',
            bg2='lb',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1)

        # Any rendering loop goes here, e.g.
        draw_scalarbar = True
        for surface_name in surface_names:
            if 'mirror' in surface_name and not plot_mirror:
                pass
            else:
                # system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.wing_vlm_mesh_out
                color = wake_color
                if 'rotor' in surface_name:
                    color = rotor_wake_color
                    var_name = 'system_model.wig.wig.wig.operation.input_model.'+surface_name[0:9]+'_rotor.' + surface_name
                if 'wing' in surface_name:
                    var_name = 'system_model.wig.wig.wig.operation.input_model.wing_vlm_meshmirror.' + surface_name
                elif 'right' in surface_name:
                    var_name = 'system_model.wig.wig.wig.operation.input_model.right_fuselage_meshmirror.' + surface_name
                elif 'left' in surface_name:
                    var_name = 'system_model.wig.wig.wig.operation.input_model.left_fuselage_meshmirror.' + surface_name
                elif 'tail' in surface_name:
                    var_name = 'system_model.wig.wig.wig.operation.input_model.htail_meshmirror.' + surface_name

                mesh_points = sim[var_name][i, :, :, :]
                nx = mesh_points.shape[0]
                ny = mesh_points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
                    vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=.5).linecolor('black')
                vp += vps
                vp += __doc__
                wake_points = sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_wake_coords'][i, 0:i, :, :]
                gamma_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_gamma_w'][i, 0:i, :], (-1,1))
                if absolute:
                    gamma_w = np.absolute(gamma_w)
                wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
                nx = wake_points.shape[0]
                ny = wake_points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
                vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity], c=color, alpha=1)
                vps.cmap(cmap, gamma_w, on='cells', vmin=min_gamma, vmax=max_gamma)
                if draw_scalarbar:
                    vps.add_scalarbar()
                    draw_scalarbar = False
                vps.linewidth(1)
                vp += vps
                vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        if side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames
