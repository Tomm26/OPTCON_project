import numpy as np

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):
    """
    LQR for LTV system with (time-varying) affine cost
    """
    # Aggiunta regolarizzazione
    reg_eps = 1e-6
    
    try:
        ns, lA = AAin.shape[1:]
    except:
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]

    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]

    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]

    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]

    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Controlli dimensionali
    if nQ != ns:
        raise ValueError("Matrix Q does not match number of states")
    if nR != ni:
        raise ValueError("Matrix R does not match number of inputs")
    if nSs != ns:
        raise ValueError("Matrix S does not match number of states")
    if nSi != ni:
        raise ValueError("Matrix S does not match number of inputs")

    if lA < TT:
        AAin = AAin.repeat(TT, axis=2)
    if lB < TT:
        BBin = BBin.repeat(TT, axis=2)
    if lQ < TT:
        QQin = QQin.repeat(TT, axis=2)
    if lR < TT:
        RRin = RRin.repeat(TT, axis=2)
    if lS < TT:
        SSin = SSin.repeat(TT, axis=2)

    augmented = False
    if qqin is not None or rrin is not None or qqfin is not None:
        augmented = True

    KK = np.zeros((ni, ns, TT))
    sigma = np.zeros((ni, TT))
    PP = np.zeros((ns, ns, TT))
    pp = np.zeros((ns, TT))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin
    
    qq = qqin if qqin is not None else np.zeros((ns, TT-1))
    rr = rrin if rrin is not None else np.zeros((ni, TT-1))
    qqf = qqfin if qqfin is not None else np.zeros(ns)

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0
    
    # Condizione terminale
    PP[:,:,-1] = QQf
    pp[:,-1] = qqf
    
    # Riccati backward
    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt] + reg_eps * np.eye(ni)  # regolarizzazione
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:, tt+1][:,None]

        # Regolarizzazione della matrice da invertire
        MMt = RRt + BBt.T @ PPtp @ BBt
        MMt = MMt + reg_eps * np.eye(MMt.shape[0])
        try:
            MMt_inv = np.linalg.inv(MMt)
        except np.linalg.LinAlgError:
            print(f"Warning: MMt not invertible at t={tt}, using pseudo-inverse")
            MMt_inv = np.linalg.pinv(MMt)

        mmt = rrt + BBt.T @ pptp
        
        PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
        ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

        # Regolarizzazione di P per stabilità numerica
        PPt = 0.5 * (PPt + PPt.T)  # simmetrizzazione
        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()

    # Forward per calcolare i guadagni
    for tt in range(TT-1):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt] + reg_eps * np.eye(ni)  # regolarizzazione
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]

        MMt = RRt + BBt.T @ PPtp @ BBt
        MMt = MMt + reg_eps * np.eye(MMt.shape[0])
        try:
            MMt_inv = np.linalg.inv(MMt)
        except np.linalg.LinAlgError:
            MMt_inv = np.linalg.pinv(MMt)

        mmt = rrt + BBt.T @ pptp

        KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
        sigma_t = -MMt_inv@mmt
        sigma[:,tt] = sigma_t.squeeze()

    # Calcola la traiettoria ottima
    for tt in range(TT - 1):
        uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
        xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]
        xx[:,tt+1] = xx_p

    return KK, sigma, PP, xx, uu