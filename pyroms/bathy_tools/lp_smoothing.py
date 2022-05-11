"""
This code is adapted from the matlab code "LP Bathymetry" by Mathieu
Dutour Sikiric (http://drobilica.irb.hr/~mathieu/Bathymetry/index.html).
For a description of the method, see

M. Dutour Sikiric, I. Janekovic, M. Kuzmic, A new approach to bathymetry
smoothing in sigma-coordinate ocean models, Ocean Modelling 29 (2009)
128--136.

*** Some modifications done by Chuning Wang

The MATLAB LP Bathymetry package is a very old package dated back to
2021. The computation involves a lot of loops and is not vectornized for
MATLAB. This makes it extremely slow for large grids.

The python code inherited all its disadvantages, and when I tested the
code it takes more than one day to process the IJS values when smoothing
a 1000x1000 grid. This is unbearable. So I decided to overhaul the
smoother to make it more Pythonish.

The overhaul consists of two parts.

Part 1, rewrite the tool functions such as GetIJS_**. Rewrite the loops
in the code using numpy array operations. The Neighborhood finder
function is also rewritten, but the 'ConnectedComponent' function is
untouched.

Part 2, update the linear programming solver. The original linear
programming solver, lpsolve55 keeps crashing when I run the code. Also,
compile and installing the package can be a problem for new Python users.
In order to simplify it (and also take advantage of recent development)
in python scientific computation, I replaced lpsolve55 with the scipy
linear programming solver (scipy.optimize.linprog). linprog is pretty
fast and support sparse matrix, which is memory conserving.

Since I cannot run lpsolve on my machine I don't know if the two solvers
produce the exact same result. But the smoothed bathymetry looks
reasonable, and it also satisfies the rx0 criterion, which I consider a
success.

2021/9/14
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from xarray import DataArray
from . import util
from . import smoothing


def LP_smoothing_rx0(MSK, Hobs, rx0max, SignConst=0, AmpConst=1.e5,
                     verbose=True):
    """
    This program perform a linear programming method in order to
    optimize the bathymetry for a fixed factor r.
    The inequality |H(e)-H(e')| / (H(e)-H(e')) <= r where H(e)=h(e)+dh(e)
    can be rewritten as two linear inequalities on dh(e) and dh(e').
    The optimal bathymetry is obtain by minimising the perturbation
    P = sum_e(|dh(e)| under the above inequalitie constraintes.

    Usage:
    Hnew = LP_smoothing_rx0(MSK, Hobs, rx0max, SignConst, AmpConst)

    ---MSK(eta_rho,xi_rho) is the mask of the grd
         1 for sea
         0 for land
    ---Hobs(eta_rho,xi_rho) is the raw depth of the grid
    ---rx0max is the target rx0 roughness factor
    ---SignConst, single value or (eta_rho,xi_rho) matrix of 0, +1, -1
         +1  only bathymetry increase are allowed.
         -1  only bathymetry decrease are allowed.
         0   increase and decrease are allowed.
         (put 0 if you are indifferent)
    ---AmpConst, single value or (eta_rho,xi_rho)  matrix of reals.
         coefficient alpha such that the new bathymetry should
         satisfy to  |h^{new} - h^{raw}| <= alpha h^{raw}
         (put 10000 if you are indifferent)
    """

    use_xarray = False
    if isinstance(Hobs, DataArray):
        use_xarray = True
        coords = Hobs.coords
        Hobs = Hobs.values
    if isinstance(MSK, DataArray):
        MSK = MSK.values

    # Get the I, J indices and coefficients for the solver
    eta_rho, xi_rho = MSK.shape

    iList, jList, sList, Constant = \
        GetIJS_rx0(MSK, Hobs, rx0max, verbose=verbose)

    iListApp, jListApp, sListApp, ConstantApp = \
        GetIJS_maxamp(MSK, Hobs, AmpConst, verbose=verbose)

    iList, jList, sList, Constant = \
        MergeIJS_listings(iList, jList, sList, Constant,
                          iListApp, jListApp, sListApp,
                          ConstantApp)

    iListApp, jListApp, sListApp, ConstantApp = \
        GetIJS_signs(MSK, SignConst, verbose=verbose)

    iList, jList, sList, Constant = \
        MergeIJS_listings(iList, jList, sList, Constant,
                          iListApp, jListApp, sListApp,
                          ConstantApp)

    TotalNbVert = int(MSK.sum())

    ObjectiveFct = np.zeros((2*TotalNbVert))
    for iVert in range(TotalNbVert):
        ObjectiveFct[TotalNbVert+iVert] = 1

    iList = iList.astype(int)
    jList = jList.astype(int)

    # Build the sparse matrix and solver
    shp = (int(len(jList)/2), 2*TotalNbVert)
    mtx = coo_matrix((sList, (iList-1, jList-1)), shape=shp)
    print('Solver Matrix Size, %d5, %d5' % shp)
    solver = linprog(c=ObjectiveFct, A_ub=mtx, b_ub=Constant,
                     options={'sparse': True})

    if not solver['success']:
        raise ValueError('Linear programming solver failed.')
    hcorr = solver['x']

    Hnew = Hobs.copy()
    Hnew[MSK == 1] = Hobs[MSK == 1] + hcorr[:TotalNbVert]
    MaxRx0 = util.roughness0(Hnew, MSK).max()
    if verbose:
        print('rx0max = ', rx0max, '  MaxRx0 = ', MaxRx0)

    if use_xarray:
        Hnew = DataArray(Hnew, coords)

    return Hnew


def LP_smoothing_rx0_heuristic(MSK, Hobs, rx0max, SignConst=0, AmpConst=1.e5):
    """
    This program perform a linear programming method in order to
    optimize the bathymetry for a fixed factor r.
    The inequality |H(e)-H(e')| / (H(e)-H(e')) <= r where H(e)=h(e)+dh(e)
    can be rewritten as two linear inequalities on dh(e) and dh(e').
    The optimal bathymetry is obtain by minimising the perturbation
    P = sum_e(|dh(e)| under the above inequalitie constraintes.
    In order to reduce the computation time, an heuristic method is
    used.

    Usage:
    Hnew = LP_smoothing_rx0_heuristic(MSK, Hobs, rx0max, SignConst,
                                          AmpConst)

    ---MSK(eta_rho,xi_rho) is the mask of the grd
         1 for sea
         0 for land
    ---Hobs(eta_rho,xi_rho) is the raw depth of the grid
    ---rx0max is the target rx0 roughness factor
    ---SignConst, single value or (eta_rho,xi_rho) matrix of 0, +1, -1
         +1  only bathymetry increase are allowed.
         -1  only bathymetry decrease are allowed.
         0   increase and decrease are allowed.
         (put 0 if you are indifferent)
    ---AmpConst, single value or (eta_rho,xi_rho)  matrix of reals.
         coefficient alpha such that the new bathymetry should
         satisfy to  |h^{new} - h^{raw}| <= alpha h^{raw}
         (put 10000 if you are indifferent)
    """

    use_xarray = False
    if isinstance(Hobs, DataArray):
        use_xarray = True
        coords = Hobs.coords
        Hobs = Hobs.values
    if isinstance(MSK, DataArray):
        MSK = MSK.values

    # the points that need to be modified
    MSKbad = GetBadPoints(MSK, Hobs, rx0max)

    eta_rho, xi_rho = MSK.shape

    Kdist = 5

    Kbad = np.where(MSKbad == 1)
    nbKbad = np.size(Kbad, 1)
    ListIdx = np.zeros((eta_rho, xi_rho), dtype=np.int)
    ListIdx[Kbad] = list(range(nbKbad))

    ListEdges = []
    nbEdge = 0
    for iK in range(nbKbad):
        iEta, iXi = Kbad[0][iK], Kbad[1][iK]
        ListNeigh = Neighborhood(MSK, iEta, iXi, 2*Kdist+1)
        bidx = np.where(MSKbad[ListNeigh[:, 0], ListNeigh[:, 1]] == 1)
        idx = ListIdx[ListNeigh[bidx, 0], ListNeigh[bidx, 1]]
        idx = idx[idx > iK]
        for iidx in idx:
            ListEdges.append([iK, iidx])
        nbEdge = nbEdge + len(idx)

    ListEdges = np.array(ListEdges)
    ListVertexStatus = ConnectedComponent(ListEdges, nbKbad)
    nbColor = int(ListVertexStatus.max())

    Hnew = Hobs.copy()
    print('-------------------------------------------------------')
    print('Total iColor, %10d', nbColor)
    for iColor in range(1, nbColor+1):
        MSKcolor = np.zeros((eta_rho, xi_rho))
        K = np.where(ListVertexStatus == iColor)
        nbK = np.size(K, 1)
        print('iColor = %10d, nbK = %6d' % (iColor, nbK), end='\r')
        for iVertex in range(nbKbad):
            if (ListVertexStatus[iVertex] == iColor):
                iEta, iXi = Kbad[0][iVertex], Kbad[1][iVertex]
                MSKcolor[iEta, iXi] = 1
                ListNeigh = Neighborhood(MSK, iEta, iXi, Kdist)
                nbNeigh = np.size(ListNeigh, 0)
                for iNeigh in range(nbNeigh):
                    iEtaN, iXiN = ListNeigh[iNeigh]
                    MSKcolor[iEtaN, iXiN] = 1
        K = np.where(MSKcolor == 1)
        MSKHobs = np.zeros((eta_rho, xi_rho))
        MSKHobs[K] = Hobs[K].copy()
        HnewI = \
            LP_smoothing_rx0(MSKcolor, MSKHobs, rx0max, SignConst, AmpConst,
                             verbose=False)
        Hnew[K] = HnewI[K]

    MaxRx0 = util.roughness0(Hnew, MSK).max()
    print('Final obtained bathymetry')
    print('rx0max = ', rx0max, '  MaxRx0 = ', MaxRx0)

    if use_xarray:
        Hnew = DataArray(Hnew, coords)

    return Hnew


def GetIJS_rx0(MSK, DEP, r, verbose=True):

    eta_rho, xi_rho = DEP.shape

    ListCoord = np.zeros((eta_rho, xi_rho))
    iidx, jidx = np.where(MSK == 1)
    ListCoord[iidx, jidx] = np.arange(len(iidx)) + 1

    TotalNbVert = len(iidx)

    if verbose:
        print('eta_rho = ', eta_rho, '  xi_rho = ', xi_rho)
        print('ListCoord built')
        print('Computing inequalities for rx0 = ', r)

    MSK2 = MSK[:-1, :] + MSK[1:, :]
    TotalNbConstant = 2*((MSK2 == 2).sum())
    TotalNbEntry = 4*((MSK2 == 2).sum())

    MSK2 = MSK[:, :-1] + MSK[:, 1:]
    TotalNbConstant = TotalNbConstant + 2*((MSK2 == 2).sum())
    TotalNbEntry = TotalNbEntry + 4*((MSK2 == 2).sum())

    TotalNbConstant = TotalNbConstant + 2 * TotalNbVert
    TotalNbEntry = TotalNbEntry + 4 * TotalNbVert

    iList = np.zeros((TotalNbEntry))

    iList[::2] = np.arange(int(TotalNbEntry/2)) + 1
    iList[1::2] = iList[::2]

    MSK2 = MSK[:-1, :] + MSK[1:, :]
    CST = (1+r)*DEP[1:, :] + (-1+r)*DEP[:-1, :]
    CST2 = (1+r)*DEP[:-1, :] + (-1+r)*DEP[1:, :]

    iidx, jidx = np.where(MSK2 == 2)
    idx1 = ListCoord[iidx, jidx]
    idx2 = ListCoord[iidx+1, jidx]

    nIdx = len(iidx)

    Constant0 = np.zeros(2*nIdx)
    jList0 = np.zeros(4*nIdx)
    sList0 = np.zeros(4*nIdx)

    Constant0[0::2] = CST[iidx, jidx]
    Constant0[1::2] = CST2[iidx, jidx]
    jList0[0::4] = idx2
    jList0[1::4] = idx1
    jList0[2::4] = idx1
    jList0[3::4] = idx2
    sList0[0::4] = -r-1
    sList0[1::4] = 1-r
    sList0[2::4] = -r-1
    sList0[3::4] = 1-r

    if verbose:
        print('Inequalities for dh(iEta,iXi) and dh(iEta+1,iXi)')

    MSK2 = MSK[:, :-1] + MSK[:, 1:]
    CST = (1+r)*DEP[:, 1:] + (r-1)*DEP[:, :-1]
    CST2 = (1+r)*DEP[:, :-1] + (r-1)*DEP[:, 1:]

    iidx, jidx = np.where(MSK2 == 2)
    idx1 = ListCoord[iidx, jidx]
    idx2 = ListCoord[iidx, jidx+1]

    nIdx = len(iidx)

    Constant1 = np.zeros(2*nIdx)
    jList1 = np.zeros(4*nIdx)
    sList1 = np.zeros(4*nIdx)

    Constant1[0::2] = CST[iidx, jidx]
    Constant1[1::2] = CST2[iidx, jidx]
    jList1[0::4] = idx2
    jList1[1::4] = idx1
    jList1[2::4] = idx1
    jList1[3::4] = idx2
    sList1[0::4] = -r-1
    sList1[1::4] = 1-r
    sList1[2::4] = -r-1
    sList1[3::4] = 1-r

    if verbose:
        print('Inequalities for dh(iEta,iXi) and dh(iEta,iXi+1)')

    iidx, jidx = np.where(MSK == 1)
    idx = ListCoord[iidx, jidx]

    nIdx = len(iidx)

    Constant2 = np.zeros(2*nIdx)
    jList2 = np.zeros(4*nIdx)
    sList2 = np.zeros(4*nIdx)

    jList2[0::4] = TotalNbVert + idx
    jList2[1::4] = idx
    jList2[2::4] = TotalNbVert + idx
    jList2[3::4] = idx
    sList2[0::4] = -1
    sList2[1::4] = 1
    sList2[2::4] = -1
    sList2[3::4] = -1

    if verbose:
        print('Inequalities dh <= ad and -dh <= ad')

    Constant = np.concatenate((Constant0, Constant1, Constant2))
    jList = np.concatenate((jList0, jList1, jList2))
    sList = np.concatenate((sList0, sList1, sList2))

    return iList, jList, sList, Constant


def GetIJS_maxamp(MSK, DEP, AmpConst, verbose=True):

    if type(AmpConst) in [float, int]:
        AmpConst = np.ones(MSK.shape)*AmpConst

    eta_rho, xi_rho = DEP.shape

    ListCoord = np.zeros((eta_rho, xi_rho))
    iidx, jidx = np.where(MSK == 1)
    ListCoord[iidx, jidx] = np.arange(len(iidx)) + 1

    MSK2 = (MSK == 1) & (AmpConst < 9999)
    iidx, jidx = np.where(MSK2)
    TotalNbConstant = 2*(MSK2.sum())
    TotalNbEntry = 2*(MSK2.sum())

    Constant = np.zeros((TotalNbConstant))
    jList = np.zeros((TotalNbEntry))
    sList = np.zeros((TotalNbEntry))

    Constant[::2] = DEP[MSK2]*AmpConst[MSK2]
    Constant[1::2] = Constant[::2]
    iList = np.arange(TotalNbEntry) + 1
    jList[::2] = ListCoord[iidx, jidx]
    jList[1::2] = jList[::2]
    sList[::2] = -1
    sList[1::2] = 1

    if verbose:
        print('eta_rho = ', eta_rho, '  xi_rho = ', xi_rho)
        print('Inequalities |h^{new} - h^{old}| <= alpha h^{old}')

    return iList, jList, sList, Constant


def GetIJS_signs(MSK, SignConst, verbose=True):

    if type(SignConst) in [float, int]:
        SignConst = np.ones(MSK.shape)*SignConst

    eta_rho, xi_rho = MSK.shape

    sign_check = (SignConst == 1) | (SignConst == 0) | (SignConst == -1)
    if not np.all(sign_check):
        raise ValueError('Wrong assigning please check SignConst')

    ListCoord = np.zeros((eta_rho, xi_rho))
    iidx, jidx = np.where(MSK == 1)
    ListCoord[iidx, jidx] = np.arange(len(iidx)) + 1

    MSK2 = (MSK == 1) & (SignConst != 0)
    iidx, jidx = np.where(MSK2)
    TotalNbConstant = MSK2.sum()
    TotalNbEntry = MSK2.sum()

    Constant = np.zeros((TotalNbConstant))
    iList = np.arange(TotalNbEntry) + 1
    jList = ListCoord[iidx, jidx]
    sList = -1*SignConst[MSK2]

    if verbose:
        print('eta_rho = ', eta_rho, '  xi_rho = ', xi_rho)
        print('Inequalities dh >= 0 or dh <= 0')

    return iList, jList, sList, Constant


def MergeIJS_listings(iList1, jList1, sList1, Constant1,
                      iList2, jList2, sList2, Constant2):
    """
    Suppose we have two sets of inequalities for two linear programs with
      the same set of variables presented in sparse form. The two descriptions
      are merged.
    """

    iList = np.concatenate((iList1, iList2))
    jList = np.concatenate((jList1, jList2))
    sList = np.concatenate((sList1, sList2))
    Constant = np.concatenate((Constant1, Constant2))

    return iList, jList, sList, Constant


def GetBadPoints(MSK, DEP, rx0max):

    RetBathy = smoothing.smoothing_Positive_rx0(MSK, DEP, rx0max)
    K1 = np.where(RetBathy != DEP)

    eta_rho, xi_rho = MSK.shape
    MSKbad = np.zeros((eta_rho, xi_rho))
    MSKbad[K1] = 1

    return MSKbad


def Neighborhood(MSK, iEta, iXi, Kdist):

    eta_rho, xi_rho = MSK.shape
    dist = 1e5*np.ones((2*Kdist+3, 2*Kdist+3))
    dist[Kdist+1, Kdist+1] = 0
    MSK2 = np.zeros((2*Kdist+3, 2*Kdist+3))
    iMin = max(0, iEta-Kdist)
    iMax = min(eta_rho, iEta+Kdist+1)
    jMin = max(0, iXi-Kdist)
    jMax = min(xi_rho, iXi+Kdist+1)
    MSK2[Kdist+1-(iEta-iMin):Kdist+1+(iMax-iEta),
         Kdist+1-(iXi-jMin):Kdist+1+(jMax-iXi)] = MSK[iMin:iMax, jMin:jMax]
    MSK2[MSK2 == 0] = 1e5

    List4dir = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    while(True):
        dist0 = dist.copy()
        for iK in range(1, Kdist+1):
            for quad in ['ur', 'lr', 'll', 'ul']:
                for iiK in range(iK):
                    if quad == 'ur':
                        iEtaN = Kdist + 1 + iiK
                        iXiN = Kdist + 1 + iK - iiK
                    elif quad == 'lr':
                        iEtaN = Kdist + 1 + iK - iiK
                        iXiN = Kdist + 1 - iiK
                    elif quad == 'll':
                        iEtaN = Kdist + 1 - iiK
                        iXiN = Kdist + 1 - iK + iiK
                    elif quad == 'ul':
                        iEtaN = Kdist + 1 - iK + iiK
                        iXiN = Kdist + 1 + iiK

                    iEtaNN = iEtaN + List4dir[:, 0]
                    iXiNN = iXiN + List4dir[:, 1]
                    if MSK2[iEtaN, iXiN] == 1:
                        dist[iEtaN, iXiN] = \
                            np.min(dist[iEtaNN, iXiNN]+MSK2[iEtaNN, iXiNN])
        dist[dist > 1e5] = 1e5
        if not np.any(dist0-dist):
            break

    dist[Kdist+1, Kdist+1] = 1e5

    ListNeighRet = np.asarray(np.where(dist <= Kdist)).T
    ListNeighRet[:, 0] = ListNeighRet[:, 0] - (Kdist+1) + iEta
    ListNeighRet[:, 1] = ListNeighRet[:, 1] - (Kdist+1) + iXi

    return ListNeighRet


def ConnectedComponent(ListEdges, nbVert):
    """
    compute the vector of connected component belonging using a representation
    and an algorithm well suited for sparse graphs.
    """

    nbEdge = ListEdges.shape[0]
    ListDegree = np.zeros(nbVert, dtype=np.int)
    ListAdjacency = np.zeros((nbVert, 10000), dtype=np.int)

    for iEdge in range(nbEdge):
        eVert = ListEdges[iEdge, 0]
        fVert = ListEdges[iEdge, 1]
        eDeg = ListDegree[eVert] + 1
        fDeg = ListDegree[fVert] + 1
        ListDegree[eVert] = eDeg
        ListDegree[fVert] = fDeg
        ListAdjacency[eVert, eDeg-1] = fVert
        ListAdjacency[fVert, fDeg-1] = eVert

    MaxDeg = ListDegree.max()
    ListAdjacency = ListAdjacency[:, :MaxDeg]

    ListVertexStatus = np.zeros(nbVert)
    ListHot = np.zeros((nbVert))
    ListNotDone = np.ones((nbVert))

    iComp = 0
    while(1):
        H = np.where(ListNotDone == 1)
        nb = np.size(H, 1)
        if (nb == 0):
            break

        iComp = iComp + 1
        ListVertexStatus[H[0][0]] = iComp
        ListHot[H[0][0]] = 1
        while(1):
            H = np.where(ListHot == 1)
            ListNotDone[H] = 0
            ListNewHot = np.zeros(nbVert)
            for iH in range(np.size(H, 1)):
                eVert = H[0][iH]
                for iV in range(ListDegree[eVert]):
                    ListNewHot[ListAdjacency[eVert, iV]] = 1

            ListHot = ListNotDone * ListNewHot
            SumH = sum(ListHot)
            if (SumH == 0):
                break

            H2 = np.where(ListHot == 1)
            ListVertexStatus[H2] = iComp

    return ListVertexStatus
