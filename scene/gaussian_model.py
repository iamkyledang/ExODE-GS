#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud

from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    strip_symmetric,
    build_scaling_rotation,
)


def _skew(v):
    """
    Build skew-symmetric matrices from vectors.

    v: (N, 3)
    return: (N, 3, 3)
    """
    N = v.shape[0]
    K = torch.zeros((N, 3, 3), device=v.device, dtype=v.dtype)

    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]

    return K


def compute_rotation_from_omega(omega, tau):
    """
    Closed-form rotation ODE:

        dQ/dtau = hat(omega) Q
        Q(0) = I
        Q(tau) = exp(hat(omega) tau)

    omega: (N, 3)
    tau:   (N, 1)
    return: (N, 3, 3)
    """
    N = omega.shape[0]

    tau = tau.view(N, 1)
    phi = omega * tau

    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)
    axis = phi / (theta + 1e-8)

    K = _skew(axis)
    K2 = torch.bmm(K, K)

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    I = I.expand(N, -1, -1)

    sin_theta = torch.sin(theta).view(N, 1, 1)
    one_minus_cos = (1.0 - torch.cos(theta)).view(N, 1, 1)

    R = I + sin_theta * K + one_minus_cos * K2
    return R


def compute_translation_from_affine_ode(A, b, tau):
    """
    Closed-form affine translation ODE using augmented matrix exponential.

        dp/dtau = A p + b
        p(0) = 0

    Instead of using A^{-1}(exp(A tau) - I)b, we use:

        M = [[A, b],
             [0, 0]]

        exp(M tau) = [[exp(A tau), p(tau)],
                      [0,          1]]

    This is stable when A is singular or near zero.

    A:   (N, 3, 3)
    b:   (N, 3)
    tau: (N, 1)
    return: (N, 3)
    """
    N = A.shape[0]

    tau = tau.view(N, 1, 1)

    M = torch.zeros((N, 4, 4), device=A.device, dtype=A.dtype)
    M[:, :3, :3] = A
    M[:, :3, 3] = b

    exp_M = torch.linalg.matrix_exp(M * tau)

    p = exp_M[:, :3, 3]
    return p


def build_se3(R, t):
    """
    Build SE(3) matrices.

    R: (N, 3, 3)
    t: (N, 3)
    return: (N, 4, 4)
    """
    N = R.shape[0]

    T = torch.zeros((N, 4, 4), device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrices to quaternions.

    R: (N, 3, 3)
    return: (N, 4), stored as [w, x, y, z]
    """
    batch = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    denom_sq = torch.stack(
        [
            trace + 1.0,
            1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
            1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2],
            1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2],
        ],
        dim=-1,
    )

    s0 = torch.sqrt(torch.clamp(denom_sq[:, 0], min=1e-10)) * 2.0
    q0 = torch.stack(
        [
            0.25 * s0,
            (R[:, 2, 1] - R[:, 1, 2]) / s0,
            (R[:, 0, 2] - R[:, 2, 0]) / s0,
            (R[:, 1, 0] - R[:, 0, 1]) / s0,
        ],
        dim=-1,
    )

    s1 = torch.sqrt(torch.clamp(denom_sq[:, 1], min=1e-10)) * 2.0
    q1 = torch.stack(
        [
            (R[:, 2, 1] - R[:, 1, 2]) / s1,
            0.25 * s1,
            (R[:, 0, 1] + R[:, 1, 0]) / s1,
            (R[:, 0, 2] + R[:, 2, 0]) / s1,
        ],
        dim=-1,
    )

    s2 = torch.sqrt(torch.clamp(denom_sq[:, 2], min=1e-10)) * 2.0
    q2 = torch.stack(
        [
            (R[:, 0, 2] - R[:, 2, 0]) / s2,
            (R[:, 0, 1] + R[:, 1, 0]) / s2,
            0.25 * s2,
            (R[:, 1, 2] + R[:, 2, 1]) / s2,
        ],
        dim=-1,
    )

    s3 = torch.sqrt(torch.clamp(denom_sq[:, 3], min=1e-10)) * 2.0
    q3 = torch.stack(
        [
            (R[:, 1, 0] - R[:, 0, 1]) / s3,
            (R[:, 0, 2] + R[:, 2, 0]) / s3,
            (R[:, 1, 2] + R[:, 2, 1]) / s3,
            0.25 * s3,
        ],
        dim=-1,
    )

    qs = torch.stack([q0, q1, q2, q3], dim=1)
    idx = denom_sq.argmax(dim=-1)

    q = qs[torch.arange(batch, device=R.device), idx]
    return F.normalize(q, dim=-1)


class GaussianModel:
    def __init__(self, sh_degree: int):
        def build_covariance_from_scaling_rotation(
            scaling,
            scaling_modifier,
            rotation,
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # ---------------------------------------------------------------------
        # Standard 3DGS parameters.
        # ---------------------------------------------------------------------
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # ---------------------------------------------------------------------
        # Explicit per-Gaussian ODE parameters.
        #
        # For every Gaussian i:
        #   _ode_A[i]      : (3, 3)
        #   _ode_b[i]      : (3,)
        #   _ode_omega[i]  : (3,)
        #   _ode_kappa[i]  : (3,)
        #
        # These have the same first dimension as _xyz.
        # ---------------------------------------------------------------------
        self._ode_A = torch.empty(0)
        self._ode_b = torch.empty(0)
        self._ode_omega = torch.empty(0)
        self._ode_kappa = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    # -------------------------------------------------------------------------
    # Standard Gaussian accessors.
    # -------------------------------------------------------------------------
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self._rotation,
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # -------------------------------------------------------------------------
    # ODE initialization and forward deformation.
    # -------------------------------------------------------------------------
    def _init_ode_params(
        self,
        n_points,
        device="cuda",
        init_A_scale=1e-6,
        init_b_scale=1e-6,
        init_omega_scale=1e-6,
        init_kappa_scale=1e-6,
    ):
        """
        Initialize one explicit ODE state per Gaussian.
        """
        ode_A = init_A_scale * torch.randn(
            n_points,
            3,
            3,
            device=device,
            dtype=torch.float,
        )

        ode_b = init_b_scale * torch.randn(
            n_points,
            3,
            device=device,
            dtype=torch.float,
        )

        ode_omega = init_omega_scale * torch.randn(
            n_points,
            3,
            device=device,
            dtype=torch.float,
        )

        ode_kappa = init_kappa_scale * torch.randn(
            n_points,
            3,
            device=device,
            dtype=torch.float,
        )

        self._ode_A = nn.Parameter(ode_A.requires_grad_(True))
        self._ode_b = nn.Parameter(ode_b.requires_grad_(True))
        self._ode_omega = nn.Parameter(ode_omega.requires_grad_(True))
        self._ode_kappa = nn.Parameter(ode_kappa.requires_grad_(True))

    def _check_ode_shape(self):
        """
        Ensure Gaussian tensors and ODE tensors are aligned.
        """
        n = self.get_xyz.shape[0]

        assert self._ode_A.shape[0] == n, (
            f"_ode_A has {self._ode_A.shape[0]} rows, but xyz has {n}"
        )
        assert self._ode_b.shape[0] == n, (
            f"_ode_b has {self._ode_b.shape[0]} rows, but xyz has {n}"
        )
        assert self._ode_omega.shape[0] == n, (
            f"_ode_omega has {self._ode_omega.shape[0]} rows, but xyz has {n}"
        )
        assert self._ode_kappa.shape[0] == n, (
            f"_ode_kappa has {self._ode_kappa.shape[0]} rows, but xyz has {n}"
        )

    def compute_ode_deformation(self, tau, is_6dof=False):
        """
        Compute explicit per-Gaussian ODE deformation.

        tau: (N, 1)

        Returns:
            d_xyz:
                if is_6dof=True:  (N, 4, 4) SE(3) matrices
                else:             (N, 3) translation offsets

            d_rotation:
                (N, 4) additive quaternion delta

            d_scaling:
                (N, 3) additive log-scale delta
        """
        self._check_ode_shape()

        n = self.get_xyz.shape[0]
        tau = tau.view(n, 1)

        A = self._ode_A
        b = self._ode_b
        omega = self._ode_omega
        kappa = self._ode_kappa

        # Translation ODE:
        #   dp/dtau = A p + b, p(0) = 0
        p = compute_translation_from_affine_ode(A, b, tau)

        # Rotation ODE:
        #   dQ/dtau = hat(omega) Q, Q(0) = I
        Q = compute_rotation_from_omega(omega, tau)

        # Scale ODE:
        #   ds/dtau = kappa
        d_scaling = kappa * tau

        if is_6dof:
            d_xyz = build_se3(Q, p)
        else:
            d_xyz = p

        q_tau = rotation_matrix_to_quaternion(Q)

        identity_q = q_tau.new_zeros(n, 4)
        identity_q[:, 0] = 1.0

        d_rotation = q_tau - identity_q

        return d_xyz, d_rotation, d_scaling

    # -------------------------------------------------------------------------
    # Creation and training setup.
    # -------------------------------------------------------------------------
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5

        fused_point_cloud = torch.tensor(
            np.asarray(pcd.points)
        ).float().cuda()

        fused_color = RGB2SH(
            torch.tensor(np.asarray(pcd.colors)).float().cuda()
        )

        features = torch.zeros(
            (
                fused_color.shape[0],
                3,
                (self.max_sh_degree + 1) ** 2,
            )
        ).float().cuda()

        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1),
                dtype=torch.float,
                device="cuda",
            )
        )

        self._xyz = nn.Parameter(
            fused_point_cloud.requires_grad_(True)
        )

        self._features_dc = nn.Parameter(
            features[:, :, 0:1]
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._features_rest = nn.Parameter(
            features[:, :, 1:]
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self._init_ode_params(fused_point_cloud.shape[0], device="cuda")

        self.max_radii2D = torch.zeros(
            (self.get_xyz.shape[0]),
            device="cuda",
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1),
            device="cuda",
        )

        self.denom = torch.zeros(
            (self.get_xyz.shape[0], 1),
            device="cuda",
        )

        self.spatial_lr_scale = 5

        # Use deform_lr if it exists; otherwise use a safe default.
        ode_lr = getattr(training_args, "ode_lr", None)
        if ode_lr is None:
            ode_lr = getattr(training_args, "deform_lr", 1e-4)

        ode_A_lr = getattr(training_args, "ode_A_lr", ode_lr)
        ode_b_lr = getattr(training_args, "ode_b_lr", ode_lr)
        ode_omega_lr = getattr(training_args, "ode_omega_lr", ode_lr)
        ode_kappa_lr = getattr(training_args, "ode_kappa_lr", ode_lr)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._ode_A],
                "lr": ode_A_lr,
                "name": "ode_A",
            },
            {
                "params": [self._ode_b],
                "lr": ode_b_lr,
                "name": "ode_b",
            },
            {
                "params": [self._ode_omega],
                "lr": ode_omega_lr,
                "name": "ode_omega",
            },
            {
                "params": [self._ode_kappa],
                "lr": ode_kappa_lr,
                "name": "ode_kappa",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """
        Learning rate scheduling per step.

        Currently only schedules xyz, matching the original code.
        ODE learning rates are constant unless you add another scheduler.
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # -------------------------------------------------------------------------
    # Saving and loading.
    # -------------------------------------------------------------------------
    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]

        for i in range(
            self._features_dc.shape[1] * self._features_dc.shape[2]
        ):
            l.append("f_dc_{}".format(i))

        for i in range(
            self._features_rest.shape[1] * self._features_rest.shape[2]
        ):
            l.append("f_rest_{}".format(i))

        l.append("opacity")

        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))

        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))

        # Explicit ODE attributes.
        for i in range(9):
            l.append("ode_A_{}".format(i))

        for i in range(3):
            l.append("ode_b_{}".format(i))

        for i in range(3):
            l.append("ode_omega_{}".format(i))

        for i in range(3):
            l.append("ode_kappa_{}".format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        self._check_ode_shape()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        ode_A = self._ode_A.detach().reshape(-1, 9).cpu().numpy()
        ode_b = self._ode_b.detach().cpu().numpy()
        ode_omega = self._ode_omega.detach().cpu().numpy()
        ode_kappa = self._ode_kappa.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_attributes()
        ]

        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                scale,
                rotation,
                ode_A,
                ode_b,
                ode_omega,
                ode_kappa,
            ),
            axis=1,
        )

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points

        plydata = PlyData.read(path)
        vertex = plydata.elements[0]

        property_names = [p.name for p in vertex.properties]

        xyz = np.stack(
            (
                np.asarray(vertex["x"]),
                np.asarray(vertex["y"]),
                np.asarray(vertex["z"]),
            ),
            axis=1,
        )

        opacities = np.asarray(vertex["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"])

        extra_f_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("f_rest_")
        ]

        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vertex[attr_name])

        features_extra = features_extra.reshape(
            (
                features_extra.shape[0],
                3,
                (self.max_sh_degree + 1) ** 2 - 1,
            )
        )

        scale_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("scale_")
        ]

        scales = np.zeros((xyz.shape[0], len(scale_names)))

        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])

        rot_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("rot")
        ]

        rots = np.zeros((xyz.shape[0], len(rot_names)))

        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(
                xyz,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )

        self._features_dc = nn.Parameter(
            torch.tensor(
                features_dc,
                dtype=torch.float,
                device="cuda",
            )
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._features_rest = nn.Parameter(
            torch.tensor(
                features_extra,
                dtype=torch.float,
                device="cuda",
            )
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._opacity = nn.Parameter(
            torch.tensor(
                opacities,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )

        self._scaling = nn.Parameter(
            torch.tensor(
                scales,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )

        self._rotation = nn.Parameter(
            torch.tensor(
                rots,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )

        n_points = xyz.shape[0]

        has_ode_A = all("ode_A_{}".format(i) in property_names for i in range(9))
        has_ode_b = all("ode_b_{}".format(i) in property_names for i in range(3))
        has_ode_omega = all(
            "ode_omega_{}".format(i) in property_names for i in range(3)
        )
        has_ode_kappa = all(
            "ode_kappa_{}".format(i) in property_names for i in range(3)
        )

        if has_ode_A and has_ode_b and has_ode_omega and has_ode_kappa:
            ode_A = np.zeros((n_points, 9))
            ode_b = np.zeros((n_points, 3))
            ode_omega = np.zeros((n_points, 3))
            ode_kappa = np.zeros((n_points, 3))

            for i in range(9):
                ode_A[:, i] = np.asarray(vertex["ode_A_{}".format(i)])

            for i in range(3):
                ode_b[:, i] = np.asarray(vertex["ode_b_{}".format(i)])
                ode_omega[:, i] = np.asarray(vertex["ode_omega_{}".format(i)])
                ode_kappa[:, i] = np.asarray(vertex["ode_kappa_{}".format(i)])

            ode_A = ode_A.reshape(n_points, 3, 3)

            self._ode_A = nn.Parameter(
                torch.tensor(
                    ode_A,
                    dtype=torch.float,
                    device="cuda",
                ).requires_grad_(True)
            )

            self._ode_b = nn.Parameter(
                torch.tensor(
                    ode_b,
                    dtype=torch.float,
                    device="cuda",
                ).requires_grad_(True)
            )

            self._ode_omega = nn.Parameter(
                torch.tensor(
                    ode_omega,
                    dtype=torch.float,
                    device="cuda",
                ).requires_grad_(True)
            )

            self._ode_kappa = nn.Parameter(
                torch.tensor(
                    ode_kappa,
                    dtype=torch.float,
                    device="cuda",
                ).requires_grad_(True)
            )

        else:
            print(
                "[GaussianModel] No ODE parameters found in PLY. "
                "Initializing new explicit ODE parameters."
            )
            self._init_ode_params(n_points, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    # -------------------------------------------------------------------------
    # Optimizer tensor replacement / pruning / concatenation.
    # -------------------------------------------------------------------------
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_param = group["params"][0]
                stored_state = self.optimizer.state.get(old_param, None)

                new_param = nn.Parameter(tensor.requires_grad_(True))

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[old_param]

                    group["params"][0] = new_param
                    self.optimizer.state[new_param] = stored_state
                else:
                    group["params"][0] = new_param

                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(
                self.get_opacity,
                torch.ones_like(self.get_opacity) * 0.01,
            )
        )

        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new,
            "opacity",
        )

        self._opacity = optimizable_tensors["opacity"]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = old_param[mask]

            new_param = nn.Parameter(new_tensor.requires_grad_(True))

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[old_param]

                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state
            else:
                group["params"][0] = new_param

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._ode_A = optimizable_tensors["ode_A"]
        self._ode_b = optimizable_tensors["ode_b"]
        self._ode_omega = optimizable_tensors["ode_omega"]
        self._ode_kappa = optimizable_tensors["ode_kappa"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._check_ode_shape()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1

            name = group["name"]
            extension_tensor = tensors_dict[name]

            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = torch.cat((old_param, extension_tensor), dim=0)
            new_param = nn.Parameter(new_tensor.requires_grad_(True))

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (
                        stored_state["exp_avg"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )

                stored_state["exp_avg_sq"] = torch.cat(
                    (
                        stored_state["exp_avg_sq"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )

                del self.optimizer.state[old_param]

                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state
            else:
                group["params"][0] = new_param

            optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    # -------------------------------------------------------------------------
    # Densification.
    # -------------------------------------------------------------------------
    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_ode_A,
        new_ode_b,
        new_ode_omega,
        new_ode_kappa,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "ode_A": new_ode_A,
            "ode_b": new_ode_b,
            "ode_omega": new_ode_omega,
            "ode_kappa": new_ode_kappa,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._ode_A = optimizable_tensors["ode_A"]
        self._ode_b = optimizable_tensors["ode_b"]
        self._ode_omega = optimizable_tensors["ode_omega"]
        self._ode_kappa = optimizable_tensors["ode_kappa"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1),
            device="cuda",
        )

        self.denom = torch.zeros(
            (self.get_xyz.shape[0], 1),
            device="cuda",
        )

        self.max_radii2D = torch.zeros(
            (self.get_xyz.shape[0]),
            device="cuda",
        )

        self._check_ode_shape()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]

        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold,
            True,
            False,
        )

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)

        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(
            N,
            1,
            1,
        )

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )

        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(
            N,
            1,
            1,
        )

        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N,
            1,
            1,
        )

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # Children inherit explicit ODE parameters from their parent.
        new_ode_A = self._ode_A[selected_pts_mask].repeat(N, 1, 1)
        new_ode_b = self._ode_b[selected_pts_mask].repeat(N, 1)
        new_ode_omega = self._ode_omega[selected_pts_mask].repeat(N, 1)
        new_ode_kappa = self._ode_kappa[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_ode_A,
            new_ode_b,
            new_ode_omega,
            new_ode_kappa,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device="cuda",
                    dtype=bool,
                ),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold,
            True,
            False,
        )

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # Clones inherit explicit ODE parameters from their parent.
        new_ode_A = self._ode_A[selected_pts_mask]
        new_ode_b = self._ode_b[selected_pts_mask]
        new_ode_omega = self._ode_omega[selected_pts_mask]
        new_ode_kappa = self._ode_kappa[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_ode_A,
            new_ode_b,
            new_ode_omega,
            new_ode_kappa,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size

            big_points_ws = (
                self.get_scaling.max(dim=1).values > 0.1 * extent
            )

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs),
                big_points_ws,
            )

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2],
            dim=-1,
            keepdim=True,
        )

        self.denom[update_filter] += 1