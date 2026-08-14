import torch
from scipy.spatial.transform import Rotation as R
from config.config import cfg


class LatticeParam:
    def __init__(self):
        ratio = 1.0 if cfg["train"] else cfg["velocity"] / cfg["vel_max_train"]
        self.vel_max = ratio * cfg["vel_max_train"]
        self.acc_max = ratio * ratio * cfg["acc_max_train"]
        self.segment_time = cfg["sgm_time"] / ratio
        self.horizon_num = cfg["horizon_num"]
        self.vertical_num = cfg["vertical_num"]
        self.traj_num = cfg["traj_num"]
        self.horizon_fov = cfg["horizon_camera_fov"]
        self.vertical_fov = cfg["vertical_camera_fov"]
        self.horizon_anchor_fov = cfg["horizon_anchor_fov"]
        self.vertical_anchor_fov = cfg["vertical_anchor_fov"]
        self.radio_total = cfg["tail_radio_max"]

        print("---------- Param --------")
        print(f"| {'max speed':<12} = {round(self.vel_max, 1):>6} |")
        print(f"| {'max accel':<12} = {round(self.acc_max, 1):>6} |")
        print(f"| {'traj time':<12} = {round(self.segment_time, 1):>6} |")
        print(f"| {'max radio':<12} = {round(self.radio_total, 1):>6} |")
        print("-------------------------")


class LatticePrimitive(LatticeParam):
    """
    Grid index layout in image (Polar coordinate indexing: row-major, bottom-left origin;
    horizon_num=5 × vertical_num=3 = 15 primitives):
                   +----+----+----+----+----+
                   | 14 | 13 | 12 | 11 | 10 |
                   +----+----+----+----+----+
                   |  9 |  8 |  7 |  6 |  5 |
                   +----+----+----+----+----+
                   |  4 |  3 |  2 |  1 |  0 |
                   +----+----+----+----+----+
    """
    _instance = None

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        search_radio = self.radio_total / 2.0                    # inner waypoint at mid-reach
        pd_init = (self.segment_time * 0.5, self.segment_time * 0.5)

        # ---- angle grids over (i, j) ----
        direction_diff = 0 if self.horizon_num == 1 else (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num
        altitude_diff = 0 if self.vertical_num == 1 else (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num

        lattice_pos_list, lattice_angle_list, lattice_Rbp_list = [], [], []
        # Primitives: Bottom to Top, Right to Left
        for i in range(self.vertical_num):
            for j in range(self.horizon_num):
                alpha = torch.tensor(-direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff)
                beta = torch.tensor(-altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff)
                lattice_pos_list.append(torch.tensor([torch.cos(beta) * torch.cos(alpha) * search_radio,
                                                      torch.cos(beta) * torch.sin(alpha) * search_radio,
                                                      torch.sin(beta) * search_radio]))
                lattice_angle_list.append(torch.tensor([alpha, beta]))
                lattice_Rbp_list.append(torch.tensor(R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False).as_matrix()))

        N = self.traj_num
        self.lattice_pos_node = torch.stack(lattice_pos_list).to(dtype=torch.float32, device=device)   # (N, 3)
        self.lattice_angle_node = torch.stack(lattice_angle_list).to(dtype=torch.float32, device=device)  # (N, 2)
        self.lattice_Rbp_node = torch.stack(lattice_Rbp_list).to(dtype=torch.float32, device=device)   # (N, 3, 3)
        self.lattice_radio_max_node = torch.full((N,), self.radio_total, dtype=torch.float32, device=device)   # (N,)
        self.lattice_pd_init_node = torch.tensor([pd_init] * N, dtype=torch.float32, device=device)            # (N, 2)

        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    # ---- accessors ----
    def getAngleLattice(self, id=None):
        if id is not None:
            return self.lattice_angle_node[id, 0], self.lattice_angle_node[id, 1]
        else:
            return self.lattice_angle_node[:, 0], self.lattice_angle_node[:, 1]

    def getRotation(self, id=None):
        if id is not None:
            return self.lattice_Rbp_node[id]
        else:
            return self.lattice_Rbp_node

    def convert_ImageGrid_LatticeID(self, id):
        """
        Convert an image-grid index to a lattice index.

        Image grid is row-major top-left origin; lattice iterates bottom-to-top,
        right-to-left, so within each h-block the relation is `lattice_within_h =
        V·H - 1 - image_within_h`. The radio-bin index h itself is unchanged.

        Works for both scalar int and tensor ids.
        """
        V_H = self.vertical_num * self.horizon_num
        return (id // V_H) * V_H + (V_H - 1 - (id % V_H))

    def to_image_order(self, x):
        """Reorder a lattice-indexed tensor (N along dim 0) into image-grid order: reverse (i, j)
        to flip the bottom-up / right-to-left lattice into top-down / left-to-right image order.
        x shape (N, ...) → returns (N, ...) reordered."""
        return x.flip(0)

    @classmethod
    def get_instance(self):
        if self._instance is None: self._instance = self()
        return self._instance
