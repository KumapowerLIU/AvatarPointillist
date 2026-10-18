import torch


class CanoGSFeature:
    def __init__(self, binding, xyz, face_orien_mat_cano, face_scaling_cano, face_center_cano):
        self.binding = binding
        self.xyz = xyz
        self.face_orien_mat_cano = face_orien_mat_cano
        self.face_scaling_cano = face_scaling_cano
        self.face_center_cano = face_center_cano

    @property
    def get_xyz_cano(self):
        xyz_cano = torch.bmm(self.face_orien_mat_cano[self.binding], self._xyz[..., None]).squeeze(-1)
        return xyz_cano * self.face_scaling_cano[self.binding] + self.face_center_cano[self.binding]
