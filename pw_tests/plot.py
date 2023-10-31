import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show, Mesh
import numpy as np

def plot_wireframe(sim, surface_names, nt, interactive = False, plot_mirror = True):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(0, 80),
        yrange=(-100, 100),
        zrange=(-5, 10),
    )
    video = Video("rotor_test.gif", fps=10, backend='imageio')
    for i in range(2, nt - 1):
        vp = Plotter(
            bg='beige',
            bg2='lb',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1)
        # Any rendering loop goes here, e.g.:
        for surface_name in surface_names:
            if 'mirror' in surface_name and not plot_mirror:
                pass
            elif 'rotor' in surface_name:
                points = sim['system_model.wig.wig.wig.operation.input_model.'+surface_name[0:9]+'_rotor.' + surface_name][i, :, :, :]
                nx = points.shape[0]
                ny = points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])

                vps = Mesh([np.reshape(points, (-1, 3)), connectivity], c='black').wireframe()
                vp += vps
                vp += __doc__
                points = sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_wake_coords'][i, 0:i, :, :]
                nx = points.shape[0]
                ny = points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
                vps = Mesh([np.reshape(points, (-1, 3)), connectivity], c='red').wireframe()
                vp += vps
                vp += __doc__
            else:
                var_name = 'operation.' + surface_name
                points = sim[var_name][i, :, :, :]
                nx = points.shape[0]
                ny = points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
                    vps = Mesh([np.reshape(points, (-1, 3)), connectivity], c='black').wireframe()
                vp += vps
                vp += __doc__
                points = sim['operation.op_' + surface_name+'_wake_coords'][i, 0:i, :, :]
                nx = points.shape[0]
                ny = points.shape[1]
                connectivity = []
                for k in range(nx-1):
                    for j in range(ny-1):
                        connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
                vps = Mesh([np.reshape(points, (-1, 3)), connectivity], c='blue').wireframe()
                vp += vps
                vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.close_window()
    vp.close_window()
    video.close()  # merge all the recorded frames

