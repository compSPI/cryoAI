from abc import ABCMeta, abstractmethod
import torch


class SchedulerBase(metaclass=ABCMeta):
    def __init__(self):
        super(SchedulerBase, self).__init__()
        self.internal_count = 0

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def set_writer(self, writer):
        ...


class FrequencyMarcher(SchedulerBase):
    def __init__(self, model, img_sz, test_convergence_every, freq_max=-1, freq_init=15, freq_step=20,
                 epsilon_convergence=18.0):
        """
        Initialization of FrequencyMarcher.

        Parameters
        ----------
        model: torch.nn.Module
        img_sz: int
        test_convergence_every: int
        freq_max: int
            Maximum number of frequencies (radius), -1 for all frequencies
        freq_init: int
            Number of frequencies (radius) to start with
        freq_step: int
            Number of frequencies to be added at each update
        epsilon_convergence: float
            epsilon for detecting convergence
        """
        super(FrequencyMarcher, self).__init__()
        if freq_max < 0:
            self.freq_max = img_sz//2 + 1
        else:
            self.freq_max = freq_max
        self.model = model
        self.img_sz = img_sz
        self.test_convergence_every = test_convergence_every
        self.f = freq_init
        self.freq_step = freq_step
        self.epsilon_convergence = epsilon_convergence
        self.volume_old = None
        print("Frequency marcher starts at f = "+str(self.f))
        self.reached_fmax = False
        self.delta = 0.
        self.writer = None

    def set_writer(self, writer):
        """
        Link ot a writer.

        Parameters
        ----------
        writer: writer
        """
        self.writer = writer

    def update(self):
        """
        Update, test convergence and write summaries.
        """
        self.internal_count += 1
        if not self.internal_count % self.test_convergence_every:
            print("Frequency marcher tests convergence.")
            self.test_convergence(self.model)
            self.writer.add_scalar("Delta frequency marcher", self.delta, self.internal_count)
            self.writer.add_scalar("Threshold delta", self.epsilon_convergence / self.f, self.internal_count)

    def test_convergence(self, model):
        """
        Test convergence.

        Parameters
        ----------
        model: torch.nn.Module
        """
        volume = model.pred_map.make_volume(resolution=self.f).cpu().numpy()
        if not self.reached_fmax:
            if self.volume_old is not None:
                delta = self.compute_distance(volume)
                # The convergence criterion decreases with self.f
                if delta < self.epsilon_convergence / self.f:
                    print("Frequency marcher detected convergence.")
                    self.f += self.freq_step
                    if self.f >= self.freq_max:
                        print("Reached fmax.")
                        self.f = self.freq_max
                        self.reached_fmax = True
                    print("f_new = "+str(self.f))
                    print("New threshold = "+str(self.epsilon_convergence / self.f))
                    volume = model.pred_map.make_volume(resolution=self.f).cpu().numpy()
            else:
                delta = 0.
            self.volume_old = volume
        else:
            delta = 0.
        self.delta = delta

    def cut_coords_plane(self, coords):
        """
        Mask input slice.

        Parameters
        ----------
        coords: torch.tensor (S*S, dim) or (S, S, dim)

        Returns
        -------
        coords_clip: torch.tensor ((2*self.f+1)**2. dim)
        """
        dim = coords.shape[-1]
        coords_clip = coords.reshape(self.img_sz, self.img_sz, dim)
        left = max(0, self.img_sz // 2 - self.f)
        right = min(self.img_sz, self.img_sz // 2 + self.f + 1)
        coords_clip = coords_clip[left:right, left:right, :]
        coords_clip = coords_clip.reshape(-1, dim)

        self.pad_left = left
        self.pad_right = self.img_sz - right
        self.n_freq = right - left

        return coords_clip

    def pad_coords_plane(self, fplane):
        """
        Zeros-pads output slice.

        Parameters
        ----------
        fplane: torch.tensor (B, 2*self.f+1. 2*self.f+1, 2)

        Returns
        -------
        output: torch.tensor (B, S, S, 2)
        """
        output = torch.nn.functional.pad(fplane, (0, 0, self.pad_left, self.pad_right, self.pad_left, self.pad_right))
        return output

    def compute_distance(self, volume):
        """
        Computes the L2 distance between two volumes.

        Parameters
        ----------
        volume: np.array (S, S, S)

        Returns
        -------
        distance: float
        """
        distance = ((self.volume_old - volume)**2).mean()
        return distance
