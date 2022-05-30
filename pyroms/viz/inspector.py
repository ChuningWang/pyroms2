import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import ipywidgets as widgets


class ROMSSliceInteractor(widgets.VBox):
    def __init__(self, var, xpix=50, ypix=50, **kwargs):
        super().__init__()

        # Prepare data
        hpos = var.roms._pos
        xi, eta = 'xi' + hpos, 'eta' + hpos
        xN, yN = var.roms.xi.size, var.roms.eta.size
        if 'ocean_time' in var.dims:
            tN = var['ocean_time'].size
            self.has_time = True
        else:
            tN = None
            self.has_time = False
        x0, y0, t0 = int(xN/2), int(yN/2), 0
        x, y = var['x' + hpos], var['y' + hpos]
        if self.has_time:
            z = -var.roms.z.mean(dim='ocean_time')
        else:
            z = -var.roms.z
        # Load coordinates into memory
        x = x.load()
        y = y.load()
        z = z.load()

        # Get pcolor strides for 3-D plots
        xstride, ystride = int(xN/xpix), int(yN/ypix)
        xinds = list(range(0, xN, xstride)) + [xN - 1]
        yinds = list(range(0, yN, ystride)) + [yN - 1]

        self.var, self.h = var, var.h
        self.xi, self.eta = xi, eta
        self.x, self.y, self.z = x, y, z
        self.xinds, self.yinds = xinds, yinds
        self.xs, self.ys, self.ts = x0, y0, t0

        xx, yx, zx = x.isel({eta: y0}), y.isel({eta: y0}), z.isel({eta: y0})
        xy, yy, zy = x.isel({xi: x0}), y.isel({xi: x0}), z.isel({xi: x0})
        if self.has_time:
            datat = var.isel(ocean_time=t0)
        else:
            datat = var
        datat.load()
        datax = datat.isel({eta: y0})
        datay = datat.isel({xi: x0})
        zx, xx, yx = xr.broadcast(zx, xx, yx)
        zy, xy, yy = xr.broadcast(zy, xy, yy)

        self.xx, self.yx, self.xy, self.yy = xx, yx, xy, yy
        self.zx, self.zy = zx, zy
        self.datat, self.datax, self.datay = datat, datax, datay

        # Make the initial plot
        ax_kwargs = {'facecolor': 'w'}
        if 'box_aspect' in kwargs:
            ax_kwargs['box_aspect'] = kwargs.pop('box_aspect')
        if 'azim' in kwargs:
            ax_kwargs['azim'] = kwargs.pop('azim')
        if 'elev' in kwargs:
            ax_kwargs['elev'] = kwargs.pop('elev')

        plt.ioff()
        fig = plt.figure(figsize=[14, 10])
        ax = mplot3d.axes3d.Axes3D(
            fig, rect=[0.05, 0.3, 0.9, 0.69],
            auto_add_to_figure=False, computed_zorder=False,
            **ax_kwargs)
        fig.add_axes(ax)
        self.fig, self.ax = fig, ax

        fig.canvas.header_visible = False
        fig.canvas.layout.min_height = '400px'
        hmax = var.h.max()
        ax.plot_surface(
            x.isel({eta: self.yinds, xi: self.xinds}),
            y.isel({eta: self.yinds, xi: self.xinds}),
            var.h.isel({eta: self.yinds, xi: self.xinds}),
            rstride=1, cstride=1,
            cmap='gist_earth_r', vmin=0, vmax=hmax,
            edgecolor=None, zorder=0.5)

        xbry, ybry = \
            np.tile(x.isel({eta: self.yinds, xi: 0}).data, 2), \
            np.tile(y.isel({eta: self.yinds, xi: 0}).data, 2)
        xbry, ybry = xbry.reshape(2, -1), ybry.reshape(2, -1)
        zbry = np.tile(var.h.isel({eta: self.yinds, xi: 0}).data,
                       2).reshape(2, -1)
        zbry[0, :] = hmax
        ax.plot_surface(
            xbry, ybry, zbry,
            rstride=1, cstride=1,
            vmin=0, vmax=hmax,
            color='gray', edgecolor='gray',
            shade=False, zorder=0.1)

        xbry, ybry = \
            np.tile(x.isel({eta: self.yinds, xi: -1}).data, 2), \
            np.tile(y.isel({eta: self.yinds, xi: -1}).data, 2)
        xbry, ybry = xbry.reshape(2, -1), ybry.reshape(2, -1)
        zbry = np.tile(var.h.isel({eta: self.yinds, xi: -1}).data,
                       2).reshape(2, -1)
        zbry[0, :] = hmax
        ax.plot_surface(
            xbry, ybry, zbry,
            rstride=1, cstride=1,
            vmin=0, vmax=hmax,
            color='gray', edgecolor='gray',
            shade=False, zorder=0.1)

        xbry, ybry = \
            np.tile(x.isel({eta: 0, xi: self.xinds}).data, 2), \
            np.tile(y.isel({eta: 0, xi: self.xinds}).data, 2)
        xbry, ybry = xbry.reshape(2, -1), ybry.reshape(2, -1)
        zbry = np.tile(var.h.isel({eta: 0, xi: self.xinds}).data,
                       2).reshape(2, -1)
        zbry[0, :] = hmax
        ax.plot_surface(
            xbry, ybry, zbry,
            rstride=1, cstride=1,
            vmin=0, vmax=hmax,
            color='gray', edgecolor='gray',
            shade=False, zorder=0.1)

        xbry, ybry = \
            np.tile(x.isel({eta: -1, xi: self.xinds}).data, 2), \
            np.tile(y.isel({eta: -1, xi: self.xinds}).data, 2)
        xbry, ybry = xbry.reshape(2, -1), ybry.reshape(2, -1)
        zbry = np.tile(var.h.isel({eta: -1, xi: self.xinds}).data,
                       2).reshape(2, -1)
        zbry[0, :] = hmax
        ax.plot_surface(
            xbry, ybry, zbry,
            rstride=1, cstride=1,
            vmin=0, vmax=hmax,
            color='gray', edgecolor='gray',
            shade=False, zorder=0.1)

        self.pcx = self.xpcolor3d(
            xx.isel({xi: self.xinds}), yx.isel({xi: self.xinds}),
            zx.isel({xi: self.xinds}), datax.isel({xi: self.xinds}),
            **kwargs)
        self.pcy = self.xpcolor3d(
            xy.isel({eta: self.yinds}), yy.isel({eta: self.yinds}),
            zy.isel({eta: self.yinds}), datay.isel({eta: self.yinds}),
            **kwargs)

        ax.set_zlim(hmax, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Pcolormesh plots
        self.axx = fig.add_subplot(427, facecolor='silver')
        self.axx.set_xlim(xx.min(), xx.max())
        self.axx.set_ylim(var.h.max(), 0)
        self.axx.set_xlabel('X')
        self.axx.set_ylabel('Z')
        self.pcmx = self.axx.pcolormesh(xx, zx, datax, shading='gouraud',
                                        **kwargs)

        self.axy = fig.add_subplot(428, facecolor='sliver', sharey=self.axx)
        self.axy.set_xlim(yy.min(), yy.max())
        self.axy.set_xlabel('Y')
        self.pcmy = self.axy.pcolormesh(yy, zy, datay, shading='gouraud',
                                        **kwargs)

        self.output = widgets.Output()
        with self.output:
            self.fig.show()

        # Widgets design - Sliders
        int_sliderx = widgets.IntSlider(
            value=x0, min=1, max=xN, step=1,
            description='Xi', continuous_update=True)
        int_slidery = widgets.IntSlider(
            value=y0, min=1, max=yN, step=1,
            description='Eta', continuous_update=True)
        # Widgets design - Text Box
        boxt = widgets.BoundedIntText(
            value=t0, min=0, max=tN,
            description='Time Step', continuous_update=False)

        # Widgets linker
        int_sliderx.observe(self.updatey, 'value')
        int_slidery.observe(self.updatex, 'value')
        boxt.observe(self.updatet, 'value')

        # Widget layouts
        controls = widgets.HBox([int_sliderx, int_slidery, boxt])
        out_box = widgets.Box([self.output])
        controls.layout = self.make_box_layout()
        out_box.layout = self.make_box_layout()

        # Add widgets to class
        self.children = [controls, out_box]

    def xpcolor3d(self, X, Y, Z, C, edgecolor=None,
                  norm=None, vmin=None, vmax=None, **kwargs):
        polys, C = self.cal_verts(X, Y, Z, C)
        polyc = mplot3d.art3d.Poly3DCollection(polys, zorder=2.5, **kwargs)
        polyc.set_array(C.ravel())
        if vmin is not None or vmax is not None:
            polyc.set_clim(vmin, vmax)
        if norm is not None:
            polyc.set_norm(norm)
        self.ax.add_collection(polyc)
        self.ax.auto_scale_xyz(
            polys[:, :, 0], polys[:, :, 1], polys[:, :, 2],
            self.ax.has_data())
        return polyc

    def cal_verts(self, X, Y, Z, C):
        # Prepare data
        if X.ndim == 1 or Y.ndim == 1:
            Z, X, Y = xr.broadcast(Z, X, Y)
        X, Y, Z, C = X.data, Y.data, Z.data, C.data

        # Process missing value.
        mask = np.isnan(X) + np.isnan(Y) + np.isnan(Z)
        mask = (mask[0:-1, 0:-1] + mask[1:, 1:] +
                mask[0:-1, 1:] + mask[1:, 0:-1])
        mask = np.isnan(C[:-1, :-1]) + mask
        unmask = ~mask

        C = np.asarray(C[:-1, :-1])[unmask]

        X1 = np.asarray(X[:-1, :-1])[unmask]
        X2 = np.asarray(X[1:, :-1])[unmask]
        X3 = np.asarray(X[1:, 1:])[unmask]
        X4 = np.asarray(X[:-1, 1:])[unmask]
        Y1 = np.asarray(Y[:-1, :-1])[unmask]
        Y2 = np.asarray(Y[1:, :-1])[unmask]
        Y3 = np.asarray(Y[1:, 1:])[unmask]
        Y4 = np.asarray(Y[:-1, 1:])[unmask]
        Z1 = np.asarray(Z[:-1, :-1])[unmask]
        Z2 = np.asarray(Z[1:, :-1])[unmask]
        Z3 = np.asarray(Z[1:, 1:])[unmask]
        Z4 = np.asarray(Z[:-1, 1:])[unmask]

        xy = np.stack([X1, Y1, Z1,
                       X2, Y2, Z2,
                       X3, Y3, Z3,
                       X4, Y4, Z4], axis=-1)
        polys = xy.reshape((-1, 4, 3))
        return polys, C

    def update_polys(self, polyc, X, Y, Z, C):
        polys, C = self.cal_verts(X, Y, Z, C)
        polyc.set_verts(polys)
        polyc.set_array(C.ravel())
        return polyc

    def update_quads(self, polyc, X, Y, C):
        Y, X = xr.broadcast(Y, X)
        h, w = X.shape
        coords = np.stack([X.data, Y.data], axis=-1)
        coords = np.asarray(coords, np.float64).reshape(h, w, 2)
        polyc._coordinates = coords
        polyc.set_paths()
        polyc.set_array(C.data)
        polyc.axes.set_xlim(X.min(), X.max())
        return polyc

    def updatet(self, change):
        if not self.has_time:
            return
        self.ts = change.new
        self.datat = self.var.isel(ocean_time=self.ts)
        self.datat.load()
        self.datax = self.datat.isel({self.eta: self.ys})
        self.datay = self.datat.isel({self.xi: self.xs})

        self.updatec(self.pcx, self.datax.isel({self.xi: self.xinds}))
        self.updatec(self.pcy, self.datay.isel({self.eta: self.yinds}))
        self.pcmx.set_array(self.datax.data)
        self.pcmy.set_array(self.datay.data)
        self.fig.canvas.draw_idle()

    def updatex(self, change):
        self.ys = change.new-1

        self.xx = self.x.isel({self.eta: self.ys})
        self.yx = self.y.isel({self.eta: self.ys})
        self.zx = self.z.isel({self.eta: self.ys})
        self.datax = self.datat.isel({self.eta: self.ys})
        self.update_polys(
            self.pcx,
            self.xx.isel({self.xi: self.xinds}),
            self.yx.isel({self.xi: self.xinds}),
            self.zx.isel({self.xi: self.xinds}),
            self.datax.isel({self.xi: self.xinds}))

        self.update_quads(self.pcmx, self.xx, self.zx, self.datax)
        self.fig.canvas.draw_idle()

    def updatey(self, change):
        self.xs = change.new-1

        self.xy = self.x.isel({self.xi: self.xs})
        self.yy = self.y.isel({self.xi: self.xs})
        self.zy = self.z.isel({self.xi: self.xs})
        self.datay = self.datat.isel({self.xi: self.xs})
        self.update_polys(
            self.pcy,
            self.xy.isel({self.eta: self.yinds}),
            self.yy.isel({self.eta: self.yinds}),
            self.zy.isel({self.eta: self.yinds}),
            self.datay.isel({self.eta: self.yinds}))

        self.update_quads(self.pcmy, self.yy, self.zy, self.datay)
        self.fig.canvas.draw_idle()

    def updatec(self, polyc, C):
        C = np.asarray(C[:-1, :-1])
        C = C[~np.isnan(C)]
        polyc.set_array(C.ravel())
        return polyc

    def make_box_layout(self):
        return widgets.Layout(
           border='solid 1px black',
           margin='0px 10px 10px 0px',
           padding='5px 5px 5px 5px')
