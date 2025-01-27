import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
import matplotlib.pyplot as plt

class TrajectoryType(Enum):
    EXPONENTIAL  = "exponential"
    CUBIC        = "cubic"
    STEP         = "step"

class TrajectoryGenerator:
    """
    Classe per generare e plottare traiettorie multi-dimensionali.
    
    Ora si assume che xPoints abbia forma (p, n) in input, 
    cioè p punti (righe) e n dimensioni (colonne).
    Nel costruttore, xPoints viene trasposta a (n, p).

    tPoints deve avere dimensione (p,):
       - tempi associati ai p punti
    T  : tempo totale
    dt : passo di campionamento
    
    Metodi di interpolazione disponibili:
        - exponential
        - cubic
        - step
    """

    def __init__(self, xPoints, tPoints, T, dt):
        # Converto in array numpy e TRASPORRE xPoints 
        # (adesso la forma diventa (n, p) internamente)
        self.xPoints = np.array(xPoints, dtype=float).T  
        self.tPoints = np.array(tPoints, dtype=float)
        self.T       = float(T)
        self.dt      = float(dt)

        # Controlli base
        if len(self.tPoints.shape) != 1:
            raise ValueError("tPoints deve essere un vettore monodimensionale.")
        if len(self.xPoints.shape) != 2:
            raise ValueError("xPoints, dopo la trasposizione, deve essere un array bidimensionale.")
        
        p1 = self.xPoints.shape[1]
        p2 = self.tPoints.shape[0]
        if p1 != p2:
            raise ValueError("Dopo la trasposizione, il numero di colonne di xPoints deve coincidere con la lunghezza di tPoints.")
        
        if self.tPoints[0] < 0 or self.tPoints[-1] > self.T:
            raise ValueError("I tempi in tPoints devono essere compresi in [0, T].")
        if self.dt <= 0:
            raise ValueError("dt deve essere positivo.")

        # Numero dimensioni e numero punti
        self.n = self.xPoints.shape[0]  # dimensioni (righe)
        self.p = self.xPoints.shape[1]  # punti noti (colonne)

    def generate_trajectory(self, traj_type=TrajectoryType.EXPONENTIAL):
        """
        Genera la traiettoria campionata (X_array, t_array) in base al tipo richiesto.
        
        Parametri:
        ----------
        traj_type : TrajectoryType
            Uno dei valori in [EXPONENTIAL, CUBIC, STEP]
        
        Ritorna:
        --------
        (X_array, t_array) : (np.ndarray, np.ndarray)
            X_array: matrice (n, N) di valori di x per ciascuna dimensione
                     (n = numero dimensioni, N = numero di campioni)
            t_array: vettore dei tempi corrispondenti (lunghezza N)
        """
        # Vettore dei tempi regolare
        t_array = np.arange(0, self.T, self.dt)
        N = len(t_array)

        # Matrice per i valori di x di ogni dimensione (n righe, N colonne)
        X_array = np.zeros((self.n, N))

        # Interpoliamo su ciascun intervallo [tPoints[j], tPoints[j+1]]
        for j in range(self.p - 1):
            t1 = self.tPoints[j]
            t2 = self.tPoints[j+1]

            idx_start = np.searchsorted(t_array, t1)
            idx_end   = np.searchsorted(t_array, t2)

            # Per ogni dimensione i
            for i in range(self.n):
                v1 = self.xPoints[i, j]
                v2 = self.xPoints[i, j+1]

                # Se i due punti coincidono, valore costante
                if v1 == v2:
                    X_array[i, idx_start:idx_end] = v1
                else:
                    # Calcoliamo la sotto-traiettoria (x_loc, t_loc)
                    if traj_type == TrajectoryType.EXPONENTIAL:
                        x_loc, _ = self._exponential_segment(t1, t2, v1, v2)
                    elif traj_type == TrajectoryType.CUBIC:
                        x_loc, _ = self._cubic_segment(t1, t2, v1, v2)
                    elif traj_type == TrajectoryType.STEP:
                        x_loc, _ = self._step_segment(t1, t2, v1, v2)
                    else:
                        raise ValueError("Tipo di traiettoria non riconosciuto.")

                    seg_len = min(idx_end - idx_start, len(x_loc))
                    X_array[i, idx_start:idx_start + seg_len] = x_loc[:seg_len]

        # Se l'ultimo tempo definito < T, proseguiamo costante con l'ultimo valore
        last_idx = np.searchsorted(t_array, self.tPoints[-1])
        for i in range(self.n):
            last_val = self.xPoints[i, -1]
            X_array[i, last_idx:] = last_val

        return X_array.T, t_array

    def plot_trajectory(self, traj_type=TrajectoryType.EXPONENTIAL, show_points=True):
        """
        Genera la traiettoria e la plotta in sottografici distinti,
        uno per ogni dimensione (riga) di xPoints.
        
        Parametri:
        ----------
        traj_type   : TrajectoryType
            Tipo di traiettoria da generare (default: EXPONENTIAL)
        show_points : bool
            Se True, mostra con scatter i punti noti per ogni dimensione.
        """
        # Generiamo la traiettoria
        X_array, t_array = self.generate_trajectory(traj_type)

        fig, axs = plt.subplots(self.n, 1, figsize=(8, 2*self.n), sharex=True)
        # Se n == 1, axs non è una lista ma un singolo oggetto
        if self.n == 1:
            axs = [axs]

        for i in range(self.n):
            ax = axs[i]
            ax.plot(t_array, X_array[i, :], label=f"Dimensione {i+1}", lw=2)

            if show_points:
                # i punti noti corrispondono a self.xPoints[i, :] 
                ax.scatter(self.tPoints, self.xPoints[i, :],
                           edgecolor='k', zorder=10, s=50, 
                           label=f"Punti noti dim {i+1}")

            ax.set_ylabel(f"x[{i}]")
            ax.grid(True)
            ax.legend()

        fig.suptitle(f"Traiettoria tipo: {traj_type.value}", fontsize=14)
        axs[-1].set_xlabel("Tempo [s]")
        plt.tight_layout()
        # plt.show(block=False)
        plt.show()

    def _exponential_segment(self, t1, t2, v1, v2):
        dt_local = self.dt
        t_loc = np.arange(t1, t2, dt_local)
        T = (t2 - t1)
        if T < 1e-8:
            return (np.array([v1]), np.array([t1]))

        k = 3.0 / T
        denom = 1.0 - np.exp(-k * T)
        x_loc = v1 + (v2 - v1) * (1 - np.exp(-k * (t_loc - t1))) / denom
        return x_loc, t_loc

    def _cubic_segment(self, t1, t2, v1, v2):
        dt_local = self.dt
        t_loc = np.arange(t1, t2, dt_local)
        if (t2 - t1) < 1e-8:
            return (np.array([v1]), np.array([t1]))
        
        spline = CubicSpline([t1, t2], [v1, v2], bc_type=((1, 0.0), (1, 0.0)))
        x_loc = spline(t_loc)
        return x_loc, t_loc

    def _step_segment(self, t1, t2, v1, v2):
        t_loc = np.arange(t1, t2, self.dt)
        if (t2 - t1) < 1e-8:
            return (np.array([v1]), np.array([t1]))

        # Rimaniamo a v1 per tutto l'intervallo [t1, t2)
        x_loc = np.full_like(t_loc, v1)
        return x_loc, t_loc


# ============== ESEMPIO D'USO ===================
if __name__ == "__main__":

    xPoints = [
        [0.0, 0.0, 0.0],   # primo punto
        [5.0, 3.0, 1.0],   # secondo punto
        [2.0, 0.0, 0.0]    # terzo punto
    ]

    tPoints = [0.0, 2.0, 4.0]

    T  = 6.0
    dt = 0.01

    gen = TrajectoryGenerator(xPoints, tPoints, T, dt)

    gen.plot_trajectory(traj_type=TrajectoryType.CUBIC)
    gen.plot_trajectory(traj_type=TrajectoryType.STEP)
    gen.plot_trajectory(traj_type=TrajectoryType.EXPONENTIAL)
