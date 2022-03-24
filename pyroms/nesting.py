import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import netCDF4 as nc
from . import grid, hgrid, vgrid


class Gcontact(object):
    """
    CONTACT:  Sets Contact Points between ROMS nested Grids.

    [S,G] = contact(Gnames, Cname, Lmask, MaskInterp, Lplot)

    This function sets contact points in the overlaping contact
    regions between nested grids. The order of nested grid file
    names in input cell array (Gnames) is important.  Set the
    file names in the order of nesting layers and time-stepping
    in ROMS.

    On Input:

       Gnames      Input Grid NetCDF file names (cell array)

       Cname       Ouptut Contact Points NetCDF file name (string)

       Lmask       Switch to remove Contact Points over land
                     (default false)

       MaskInterp  Switch to interpolate PSI-, U- and V-masks (true) or
                     computed from interpolated RHO-mask (false) using
                     the "uvp_mask" script. We highly recommend for this
                     switch to always be false (default false)

       Lplot       Switch to plot various Contact Points figures
                     (default false)

    On Output:

       S           Nested grids Contact Points structure (struct array)
       G           Nested grids structure (1 x Ngrids struct array)

    Calls to External Functions:

       get_roms_grid     Gets Information Grids Structure, G(ng)
       grid_perimeter    Sets Nested Grids Perimeters and Boundary Edges
       grid_connections  Sets Nested Grids Connectivity
       plot_contact      Plots various Contact Points figures
       write_contact     Creates and writes Contact Point data to NetCDF file

    The Contact Points structure has the following fields:

       S.Ngrids                              - Number of nested grids
       S.Ncontact                            - Number of contact regions
       S.NLweights = 4                       - Number of linear weights
       S.NQweights = 9                       - Number of quadratic weights
       S.Ndatum                              - Total number of contact points

       S.western_edge  = 1                   - Western  boundary edge index
       S.southern_edge = 2                   - Southern boundary edge index
       S.eastern_edge  = 3                   - Eastern  boundary edge index
       S.northern_edge = 4                   - Northern boundary edge index

       S.spherical                           - Spherical switch

       S.grid(ng).filename                   - Grid NetCDF file name

       S.grid(ng).Lp                         - Number of I-points (RHO)
       S.grid(ng).Mp                         - Number of J-points (RHO)
       S.grid(ng).L                          - Number of I-points (PSI)
       S.grid(ng).M                          - Number of J-points (PSI)

       S.grid(ng).refine_factor              - Refinement factor (0,3,5,7)
       S.grid(ng).parent_Imin                - Donor I-left   extract index
       S.grid(ng).parent_Imax                - Donor I-right  extract index
       S.grid(ng).parent_Jmin                - Donor J-bottom extract index
       S.grid(ng).parent_Jmax                - Donor J-top    extract index

       S.grid(ng).I_psi(:,:)                 - ROMS I-indices at PSI-points
       S.grid(ng).J_psi(:,:)                 - ROMS J-indices at PSI-points
       S.grid(ng).I_rho(:,:)                 - ROMS I-indices at RHO-points
       S.grid(ng).J_rho(:,:)                 - ROMS J-indices at RHO-points
       S.grid(ng).I_u  (:,:)                 - ROMS I-indices at U-points
       S.grid(ng).J_u  (:,:)                 - ROMS J-indices at U-points
       S.grid(ng).I_v  (:,:)                 - ROMS I-indices at V-points
       S.grid(ng).J_v  (:,:)                 - ROMS I-indices at V-points

       S.grid(ng).perimeter.X_psi(:)         - Perimeter X-coordinates (PSI)
       S.grid(ng).perimeter.Y_psi(:)         - Perimeter Y-coordinates (PSI)
       S.grid(ng).perimeter.X_rho(:)         - Perimeter X-coordinates (RHO)
       S.grid(ng).perimeter.Y_rho(:)         - Perimeter Y-coordinates (RHO)
       S.grid(ng).perimeter.X_u(:)           - Perimeter X-coordinates (U)
       S.grid(ng).perimeter.Y_u(:)           - Perimeter Y-coordinates (U)
       S.grid(ng).perimeter.X_v(:)           - Perimeter X-coordinates (V)
       S.grid(ng).perimeter.Y_v(:)           - Perimeter Y-coordinates (V)
                                               (counterclockwise)

       S.grid(ng).corners.index(:)           - Corners linear IJ-index
       S.grid(ng).corners.X(:)               - Corners X-coordinates (PSI)
       S.grid(ng).corners.Y(:)               - Corners Y-coordinates (PSI)
       S.grid(ng).corners.I(:)               - Corners I-indices (PSI)
       S.grid(ng).corners.J(:)               - Corners J-indices (PSI)

       S.grid(ng).boundary(ib).index(:)      - Boundary linear IJ-index
       S.grid(ng).boundary(ib).X(:)          - Boundary X-coordinates (PSI)
       S.grid(ng).boundary(ib).Y(:)          - Boundary Y-coordinates (PSI)
                                               (without corner points)

       S.contact(cr).donor_grid              - Donor grid number
       S.contact(cr).receiver_grid           - Receiver grid number
       S.contact(cr).coincident              - Coincident boundary switch
       S.contact(cr).composite               - Composite grid switch
       S.contact(cr).hybrid                  - Hybrid nested grids switch
       S.contact(cr).mosaic                  - Mosaic grid switch
       S.contact(cr).refinement              - Refinement grid switch

       S.contact(cr).interior.okay           - true/false logical

       S.contact(cr).interior.Xdg(:)         - (X,Y) coordinates and (I,J)
       S.contact(cr).interior.Ydg(:)           indices of donor grid points
       S.contact(cr).interior.Idg(:)           inside the receiver grid
       S.contact(cr).interior.Jdg(:)           perimeter, [] if false

       S.contact(cr).corners.okay            - true/false logical

       S.contact(cr).corners.Xdg(:)          - (X,Y) coordinates and (I,J)
       S.contact(cr).corners.Ydg(:)            indices of donor grid points
       S.contact(cr).corners.Idg(:)            corners laying on receiver
       S.contact(cr).corners.Idg(:)            grid perimeter, [] if false

       S.contact(cr).boundary(ib).okay       - true/false logical
       S.contact(cr).boundary(ib).match(:)   - Donor matching points logical

       S.contact(cr).boundary(ib).Xdg(:)     - (X,Y) coordinates and (I,J)
       S.contact(cr).boundary(ib).Ydg(:)       indices of donor boundary
       S.contact(cr).boundary(ib).Idg(:)       points laying on receiver
       S.contact(cr).boundary(ib).Jdg(:)       grid perimeter, [] if false

       S.contact(cr).point.xrg_rho(:)        - Receiver grid contact points
       S.contact(cr).point.erg_rho(:)          (XI,ETA) coordinates, (X,Y)
       S.contact(cr).point.Xrg_rho(:)          physical coordinates, and
       S.contact(cr).point.Yrg_rho(:)          (I,J) indices (RHO-points)
       S.contact(cr).point.Irg_rho(:)          where data is needed from
       S.contact(cr).point.Jrg_rho(:)          donor

       S.contact(cr).point.Idg_rho(:)        - Donor (I,J) cell containing
       S.contact(cr).point.Jdg_rho(:)          receiver grid contact point

       S.contact(cr).point.xrg_u(:)          - Receiver grid contact points
       S.contact(cr).point.erg_u(:)            (XI,ETA) coordinates, (X,Y)
       S.contact(cr).point.Xrg_u(:)            physical coordinates, and
       S.contact(cr).point.Yrg_u(:)            (I,J) indices (U-points)
       S.contact(cr).point.Irg_u(:)            where data is needed from
       S.contact(cr).point.Jrg_u(:)            donor

       S.contact(cr).point.Idg_u(:)          - Donor (I,J) cell containing
       S.contact(cr).point.Jdg_u(:)            receiver grid contact point

       S.contact(cr).point.xrg_v(:)          - Receiver grid contact points
       S.contact(cr).point.erg_v(:)            (XI,ETA) coordinates, (X,Y)
       S.contact(cr).point.Xrg_v(:)            physical coordinates, and
       S.contact(cr).point.Yrg_v(:)            (I,J) indices (V-points)
       S.contact(cr).point.Irg_v(:)            where data is needed from
       S.contact(cr).point.Jrg_v(:)            donor

       S.contact(cr).point.Idg_v(:)          - Donor (I,J) cell containing
       S.contact(cr).point.Jdg_v(:)            receiver grid contact point

       S.contact(cr).point.boundary_rho      - Contact point on RHO-boundary
       S.contact(cr).point.boundary_u        - Contact point on   U-boundary
       S.contact(cr).point.boundary_v        - Contact point on   V-boundary

                                             - Donor data at contact point:
       S.contact(cr).point.angle(:)              angle between XI-axis & East
       S.contact(cr).point.f(:)                  Coriolis parameter
       S.contact(cr).point.h(:)                  bathymetry
       S.contact(cr).point.pm(:)                 curvilinear  XI-metric, 1/dx
       S.contact(cr).point.pn(:)                 curvilinear ETA-metric, 1/dy
       S.contact(cr).point.dndx(:)               d(pn)/d(xi)  metric
       S.contact(cr).point.dmde(:)               d(pm)/d(eta) metric
       S.contact(cr).point.mask_rho(:)           Land/Sea mask at RHO-points
       S.contact(cr).point.mask_u(:)             Land/Sea mask at U-points
       S.contact(cr).point.mask_v(:)             land/Sea mask at V-contact

       S.refined(cr).xi_rho(:,:)             - Receiver grid curvilinear
       S.refined(cr).eta_rho(:,:)              (XI,ETA) coordinates

       S.refined(cr).x_rho(:,:)              - Receiver grid Cartesian (X,Y)
       S.refined(cr).y_rho(:,:)                coordinates of the contact
       S.refined(cr).x_psi(:,:)                points that requires data from
       S.refined(cr).y_psi(:,:)                donor grid for each C-grid
       S.refined(cr).x_u(:,:)                  type variable (RHO-, PSI, U-,
       S.refined(cr).y_u(:,:)                  V-points). It is used to set
       S.refined(cr).x_v(:,:)                  contact points and weights
       S.refined(cr).y_v(:,:)                  in refinement grid.

       S.refined(cr).lon_rho(:,:)            - Receiver grid spherical
       S.refined(cr).lat_rho(:,:)              (lon,lat) coordinates of the
       S.refined(cr).lon_psi(:,:)              contact points that require
       S.refined(cr).lat_psi(:,:)              data from donor grid for each
       S.refined(cr).lon_u(:,:)                C-grid type variable (RHO-,
       S.refined(cr).lat_u(:,:)                PSI, U-, and V-points). It is
       S.refined(cr).lon_v(:,:)                to set contact points and
       S.refined(cr).lat_v(:,:)                weights in refinement grids.

       S.refined(cr).Irg_rho(:,:)            - Receiver grid ROMS indices
       S.refined(cr).Jrg_rho(:,:)              (I,J) of the contact points
       S.refined(cr).Irg_psi(:,:)              that require data from donor
       S.refined(cr).Jrg_psi(:,:)              grid for each C-grid type
       S.refined(cr).Irg_u(:,:)                variable (RHO-, PSI, U-, and
       S.refined(cr).Jrg_u(:,:)                V-points). It is used to set
       S.refined(cr).Irg_v(:,:)                contact points and weights
       S.refined(cr).Jrg_v(:,:)                in refinement grids.

       S.refined(cr).mask_rho(:,:)           - Receiver grid land/sea mask
       S.refined(cr).mask_psi(:,:)             at contact points interpolated
       S.refined(cr).mask_u(:,:)               from donor grid at RHO-, PSI-,
       S.refined(cr).mask_v(:,:)               U- and V-points.

       S.Lweights(cr).H_rho(4,:)             - Linear weights (H) to
       S.Lweights(cr).H_u(4,:)                 horizontally interpolate
       S.Lweights(cr).H_v(4,:)                 reciever data from donor grid

       S.Qweights(cr).H_rho(9,:)             - Quadratic weights (H) to
       S.Qweights(cr).H_u(9,:)                 horizontally interpolate
       S.Qweights(cr).H_v(9,:)                 reciever data from donor grid

    The "refined" sub-structure is only relevant when processing the contact
    region of a refinement grid. Otherwise, it will be empty.  The setting
    of Land/Sea masking in the contact region is critical. Usually, the
    RHO-mask is interpolated from the coarser grid and the U- and V-masks
    are computed from the interpolated RHO-mask using "uvp_masks".  The
    "MaskInterp" switch can be used to either interpolate (true) their
    values or compute using "uvp_masks" (false).  Recall that we are not
    modifying the original refined grid mask, just computing the mask in
    the contact region adjacent to the finer grid from the coarser grid mask.
    This is only relevant when there are land/sea masking features in any
    of the refinement grid physical boundaries. If so, the user just needs to
    experiment with "MaskInterp" and edit such points during post-processing.

    The locations of the spatial linear interpolation weights in the donor
    grid with respect the receiver grid contact region at a particulat
    contact point x(Irg,Jrg,Krg) are:

                          8___________7   (Idg+1,Jdg+1,Kdg)
                         /.          /|
                        / .         / |
     (Idg,Jdg+1,Kdg)  5/___________/6 |
                       |  .        |  |
                       |  .   x    |  |
                       | 4.........|..|3  (Idg+1,Jdg+1,Kdg-1)
                       | .         |  /
                       |.          | /
                       |___________|/
     (Idg,Jdg,Kdg-1)   1           2

                                           Suffix:   dg = donor grid
                                                     rg = receiver grid

    We just need to set the horizontal interpolation weights 1:4. The
    Other weights needed for vertical interpolation will be set inside
    ROMS.  Notice that if the contact point "cp" between donor and
    receiver grids are coincident:

             S.Lweights(cr).H_rho(1,cp) = 1.0
             S.Lweights(cr).H_rho(2,cp) = 0.0
             S.Lweights(cr).H_rho(3,cp) = 0.0
             S.Lweights(cr).H_rho(4,cp) = 0.0
    Then
             receiver_value(Irg,Jrg) = donor_value(Idg,Jdg)
    """

    def __init__(self, gnames, lmask=False, maskInterp=False, lplot=False):
        self.gnames = gnames
        self.grd = []
        for G in gnames:
            self.grd.append(grid.get_ROMS_grid(
                'File name: ' + G, hist_file=G, grid_file=G))
        self.ngrids = len(gnames)
        self.ncontact = (self.ngrids-1)*2
        return

    def grids_structure(self):
        return


def coarse2fine(grd_c, gfactor, imin, imax, jmin, jmax,
                fname='roms_grd_f.nc', lplot=False):
    mp, lp = grd_c.hgrid.x_rho.shape
    if imin <= 0 or imax >= mp or jmin <= 0 or jmax >= lp:
        raise ValueError('i/j index exceeds coarse grid range.')
    delta = 1./gfactor
    xpc, ypc = np.arange(mp+1), np.arange(lp+1)
    xrc, yrc = 0.5*(xpc[1:]+xpc[:-1]), 0.5*(ypc[1:]+ypc[:-1])
    xpf = np.arange(imin, imax+0.1*delta, delta)
    ypf = np.arange(jmin, jmax+0.1*delta, delta)
    xrf, yrf = 0.5*(xpf[1:]+xpf[:-1]), 0.5*(ypf[1:]+ypf[:-1])

    if hasattr(grd_c.hgrid, 'lon_vert'):
        method = 'cubic'
    else:
        method = 'linear'

    itp = interp2d(xpc, ypc, grd_c.hgrid.x_vert, method)
    x_vert = itp(xpf, ypf)
    itp = interp2d(xpc, ypc, grd_c.hgrid.y_vert, method)
    y_vert = itp(xpf, ypf)
    lon_vert, lat_vert = grd_c.hgrid.proj(x_vert, y_vert, inverse=True)
    hgrd = hgrid.CGridGeo(lon_vert, lat_vert, grd_c.hgrid.proj)

    itp = interp2d(xrc, yrc, grd_c.vgrid.h, method)
    h = itp(xrf, yrf)
    if hasattr(grd_c.vgrid, 'zice'):
        itp = interp2d(xrc, yrc, grd_c.vgrid.zice, method)
        zice = itp(xrf, yrf)
    else:
        zice = np.zeros(xrf.shape)
    if hasattr(grd_c.vgrid, 'hraw'):
        itp = interp2d(xrc, yrc, grd_c.vgrid.hraw[0], method)
        hraw = itp(xrf, yrf)
    else:
        hraw = h.copy()

    if hasattr(grd_c.hgrid, 'mask_rho'):
        xrf_int, yrf_int = np.floor(xrf)+0.5, np.floor(yrf)+0.5
        itp = interp2d(xrc, yrc, grd_c.hgrid.mask_rho)
        mask_rho = itp(xrf_int, yrf_int)
        hgrd.mask_rho = mask_rho

    if hasattr(grd_c.hgrid, 'mask_is'):
        xrf_int, yrf_int = np.floor(xrf)+0.5, np.floor(yrf)+0.5
        itp = interp2d(xrc, yrc, grd_c.hgrid.mask_is)
        mask_is = itp(xrf_int, yrf_int)
        hgrd.mask_is = mask_is

        # itp = interp2d(xrc, yrc, grd_c.hgrid.mask_is, method)
        # zero_filt = itp(xrf, yrf)
        # zice = zice/zero_filt
        # zice[mask_is == 0] = 0

    vgrd = vgrid.SCoord(
        h, grd_c.vgrid.theta_b, grd_c.vgrid.theta_s,
        grd_c.vgrid.Tcline, grd_c.vgrid.N,
        Vtrans=grd_c.vgrid.Vtrans, Vstretch=grd_c.vgrid.Vstretch,
        hraw=hraw, zice=zice)

    grd_f = grid.ROMSGrid('Ross Sea fine grid', hgrd, vgrd)
    grid.write_ROMS_grid(grd_f, filename=fname)
    fh = nc.Dataset(fname, 'a')
    if gfactor == 1:
        fh.donor_grid = grd_c.name
        fh.donor_Imin = imin
        fh.donor_Imax = imax
        fh.donor_Jmin = jmin
        fh.donor_Jmax = jmax
        fh.sampling_factor = gfactor
    else:
        fh.parent_grid = grd_c.name
        fh.parent_Imin = imin
        fh.parent_Imax = imax
        fh.parent_Jmin = jmin
        fh.parent_Jmax = jmax
        fh.refine_factor = gfactor
    fh.history = 'GRID file created using pyroms.'
    fh.close()

    if lplot:
        msk_c = grd_c.hgrid.mask_rho
        msk_f = grd_f.hgrid.mask_rho
        msk_c = np.ma.masked_where(msk_c == 0, msk_c)
        msk_f = np.ma.masked_where(msk_f == 0, msk_f)
        cmap_c = ListedColormap(['gray', 'w'])
        cmap_f = ListedColormap(['r', 'w'])
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.pcolormesh(grd_c.hgrid.x_vert, grd_c.hgrid.y_vert,
                      msk_c, cmap=cmap_c, alpha=0.25)
        ax.pcolormesh(grd_f.hgrid.x_vert, grd_f.hgrid.y_vert,
                      msk_f, cmap=cmap_f, alpha=0.25)
        ax.plot(grd_c.hgrid.x_vert[0], grd_c.hgrid.y_vert[0], 'k')
        ax.plot(grd_c.hgrid.x_vert[-1], grd_c.hgrid.y_vert[-1], 'k')
        ax.plot(grd_c.hgrid.x_vert[:, 0], grd_c.hgrid.y_vert[:, 0], 'k')
        ax.plot(grd_c.hgrid.x_vert[:, -1], grd_c.hgrid.y_vert[:, -1], 'k')
        ax.plot(grd_f.hgrid.x_vert[0], grd_f.hgrid.y_vert[0], 'k')
        ax.plot(grd_f.hgrid.x_vert[-1], grd_f.hgrid.y_vert[-1], 'k')
        ax.plot(grd_f.hgrid.x_vert[:, 0], grd_f.hgrid.y_vert[:, 0], 'k')
        ax.plot(grd_f.hgrid.x_vert[:, -1], grd_f.hgrid.y_vert[:, -1], 'k')
        plt.show()

    return grd_f
