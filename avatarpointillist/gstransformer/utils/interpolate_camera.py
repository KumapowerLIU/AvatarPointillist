import torch
import torch.nn.functional as F

def select_multiview_camera_path(c2ws, intrs, loop_cycle=None):
    """
    沿着多目相机的路径走一圈，不插值，直接采样已有视角。

    Args:
        c2ws (torch.Tensor): 所有相机外参，shape = (N, 4, 4)
        intrs (torch.Tensor): 所有相机内参，shape = (N, 4, 4)
        loop_cycle (tuple or list): 每帧中要采样的相机编号顺序，默认闭环顺序

    Returns:
        tuple:
            - path_c2ws: (num_frames * len(loop_cycle), 4, 4)
            - path_intrs: same shape
            - used_indices: (num_frames * len(loop_cycle),)
    """
    CAMS_PER_FRAME = 16
    if loop_cycle is None:
        # loop_cycle = (15, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 14)
        loop_cycle = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15)

    num_total = c2ws.shape[0]
    num_frames = num_total // CAMS_PER_FRAME
    device = c2ws.device

    all_indices = []
    for f in range(num_frames):
        base = f * CAMS_PER_FRAME
        indices = [base + cid for cid in loop_cycle]
        all_indices.extend(indices)

    all_indices = torch.tensor(all_indices, dtype=torch.long, device=device)
    path_c2ws = c2ws[all_indices]
    path_intrs = intrs[all_indices]

    return path_c2ws, path_intrs, all_indices
def interpolate_multiview_cameras(c2ws, intrs, frames_per_loop: int = 128):
    """
    对多目相机参数进行插值，生成单目相机参数序列

    Args:
        c2ws (torch.Tensor): 相机外参的逆矩阵, shape (N, 4, 4)
        intrs (torch.Tensor): 相机内参, shape (N, 4, 4)
        frames_per_loop (int): 相机轨迹循环一圈所需要的帧数

    Returns:
        tuple: (res_c2ws, res_intrs, nearest_indices)
            - res_c2ws: (N // 16, 4, 4) 插值后的相机外参的逆矩阵
            - res_intrs: (N // 16, 4, 4) 插值后的相机内参
            - nearest_indices: (N // 16,) 实际使用的相机索引的最近邻索引
    """
    # 定义插值顺序（闭环路径）
    # LOOP_CYCLE = (15, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 14)
    LOOP_CYCLE = (  0, 2, 4, 6,8,10,12,15)
    CAMS_PER_FRAME = 8

    # 计算帧数
    num_frames = c2ws.shape[0] // CAMS_PER_FRAME
    device = c2ws.device
    dtype = c2ws.dtype

    # 预计算所有插值参数
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
    idx_rates = (frame_indices / frames_per_loop) * CAMS_PER_FRAME
    idxs = idx_rates.long()
    rates = idx_rates - idxs  # (num_frames,)

    # 处理循环边界
    idxs = idxs % len(LOOP_CYCLE)
    next_idxs = (idxs + 1) % len(LOOP_CYCLE)

    # 获取对应的相机索引
    cam_indices1 = torch.tensor([LOOP_CYCLE[i] for i in idxs.tolist()], device=device)
    cam_indices2 = torch.tensor([LOOP_CYCLE[i] for i in next_idxs.tolist()], device=device)

    # 计算实际的数组索引
    base_frame_indices = torch.arange(num_frames, device=device) * CAMS_PER_FRAME
    actual_indices1 = base_frame_indices + cam_indices1
    actual_indices2 = base_frame_indices + cam_indices2

    # 批量提取所有需要的相机参数
    c2w1_batch = c2ws[actual_indices1]  # (num_frames, 4, 4)
    c2w2_batch = c2ws[actual_indices2]  # (num_frames, 4, 4)
    intr1_batch = intrs[actual_indices1]  # (num_frames, 4, 4)
    intr2_batch = intrs[actual_indices2]  # (num_frames, 4, 4)

    # 批量插值
    res_c2ws = _batch_c2w_interpolate(c2w1_batch, c2w2_batch, rates)
    res_intrs = _batch_intr_interpolate(intr1_batch, intr2_batch, rates)

    # 实际使用的相机索引的最近邻索引
    nearest_indices = torch.where(rates < 0.5, actual_indices1, actual_indices2)

    return res_c2ws, res_intrs, nearest_indices

def _batch_c2w_interpolate(c2w1_batch, c2w2_batch, rates):
    """
    批量相机外参的逆矩阵插值（使用四元数球面插值）

    Args:
        c2w1_batch (torch.Tensor): 外参的逆矩阵批次1, shape (B, 4, 4)
        c2w2_batch (torch.Tensor): 外参的逆矩阵批次2, shape (B, 4, 4)
        rates (torch.Tensor): 插值比率, shape (B,)

    Returns:
        torch.Tensor: 插值结果, shape (B, 4, 4)
    """
    batch_size = c2w1_batch.shape[0]
    device = c2w1_batch.device
    dtype = c2w1_batch.dtype

    # 分离旋转和平移
    rot1 = c2w1_batch[:, :3, :3]  # (B, 3, 3)
    trans1 = c2w1_batch[:, :3, 3]  # (B, 3)
    rot2 = c2w2_batch[:, :3, :3]  # (B, 3, 3)
    trans2 = c2w2_batch[:, :3, 3]  # (B, 3)

    # 旋转矩阵转四元数
    quat1 = _batch_matrix_to_quaternion(rot1)  # (B, 4)
    quat2 = _batch_matrix_to_quaternion(rot2)  # (B, 4)

    # 球面线性插值
    interpolated_quat = _batch_slerp(quat1, quat2, rates)  # (B, 4)

    # 四元数转旋转矩阵
    interpolated_rot = _batch_quaternion_to_matrix(interpolated_quat)  # (B, 3, 3)

    # 线性插值平移
    rates_expanded = rates.unsqueeze(-1)  # (B, 1)
    interpolated_trans = trans1 + rates_expanded * (trans2 - trans1)  # (B, 3)

    # 组合结果
    result = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
    result[:, :3, :3] = interpolated_rot
    result[:, :3, 3] = interpolated_trans
    result[:, 3, 3] = 1.0

    return result

def _batch_intr_interpolate(intr1_batch, intr2_batch, rates):
    """
    批量相机内参线性插值

    Args:
        intr1_batch (torch.Tensor): 内参批次1, shape (B, 4, 4)
        intr2_batch (torch.Tensor): 内参批次2, shape (B, 4, 4)
        rates (torch.Tensor): 插值比率, shape (B,)

    Returns:
        torch.Tensor: 插值结果, shape (B, 4, 4)
    """
    rates_expanded = rates.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    return intr1_batch + rates_expanded * (intr2_batch - intr1_batch)

def _batch_matrix_to_quaternion(matrices):
    """
    批量3x3旋转矩阵转四元数 [w, x, y, z]

    Args:
        matrices (torch.Tensor): 旋转矩阵批次, shape (B, 3, 3)

    Returns:
        torch.Tensor: 四元数批次, shape (B, 4)
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype

    # 计算迹
    trace = torch.einsum('bii->b', matrices)  # (B,)

    # 初始化四元数
    quats = torch.empty(batch_size, 4, device=device, dtype=dtype)

    # 处理 trace > 0 的情况
    cond_positive = trace > 0
    if cond_positive.any():
        indices_pos = torch.where(cond_positive)[0]
        mats_pos = matrices[indices_pos]
        traces_pos = trace[indices_pos]

        s = torch.sqrt(traces_pos + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (mats_pos[:, 2, 1] - mats_pos[:, 1, 2]) / s
        qy = (mats_pos[:, 0, 2] - mats_pos[:, 2, 0]) / s
        qz = (mats_pos[:, 1, 0] - mats_pos[:, 0, 1]) / s

        quats[indices_pos, 0] = qw
        quats[indices_pos, 1] = qx
        quats[indices_pos, 2] = qy
        quats[indices_pos, 3] = qz

    # 处理 trace <= 0 的情况
    cond_negative = ~cond_positive
    if cond_negative.any():
        indices_neg = torch.where(cond_negative)[0]
        mats_neg = matrices[indices_neg]

        # 找到最大的对角元素
        diag_elements = torch.stack([
            mats_neg[:, 0, 0], mats_neg[:, 1, 1], mats_neg[:, 2, 2]
        ], dim=-1)  # (num_neg, 3)
        max_indices = torch.argmax(diag_elements, dim=-1)  # (num_neg,)

        # 根据最大元素选择不同的计算方式
        for i in range(3):
            mask = (max_indices == i)
            if mask.any():
                sub_indices = indices_neg[mask]
                mats_sub = mats_neg[mask]

                if i == 0:  # m[0,0] 最大
                    s = torch.sqrt(1.0 + mats_sub[:, 0, 0] - mats_sub[:, 1, 1] - mats_sub[:, 2, 2]) * 2
                    qw = (mats_sub[:, 2, 1] - mats_sub[:, 1, 2]) / s
                    qx = 0.25 * s
                    qy = (mats_sub[:, 0, 1] + mats_sub[:, 1, 0]) / s
                    qz = (mats_sub[:, 0, 2] + mats_sub[:, 2, 0]) / s
                elif i == 1:  # m[1,1] 最大
                    s = torch.sqrt(1.0 + mats_sub[:, 1, 1] - mats_sub[:, 0, 0] - mats_sub[:, 2, 2]) * 2
                    qw = (mats_sub[:, 0, 2] - mats_sub[:, 2, 0]) / s
                    qx = (mats_sub[:, 0, 1] + mats_sub[:, 1, 0]) / s
                    qy = 0.25 * s
                    qz = (mats_sub[:, 1, 2] + mats_sub[:, 2, 1]) / s
                else:  # m[2,2] 最大
                    s = torch.sqrt(1.0 + mats_sub[:, 2, 2] - mats_sub[:, 0, 0] - mats_sub[:, 1, 1]) * 2
                    qw = (mats_sub[:, 1, 0] - mats_sub[:, 0, 1]) / s
                    qx = (mats_sub[:, 0, 2] + mats_sub[:, 2, 0]) / s
                    qy = (mats_sub[:, 1, 2] + mats_sub[:, 2, 1]) / s
                    qz = 0.25 * s

                quats[sub_indices, 0] = qw
                quats[sub_indices, 1] = qx
                quats[sub_indices, 2] = qy
                quats[sub_indices, 3] = qz

    # 标准化四元数
    return F.normalize(quats, p=2, dim=-1)

def _batch_quaternion_to_matrix(quaternions):
    """
    批量四元数转3x3旋转矩阵

    Args:
        quaternions (torch.Tensor): 四元数批次, shape (B, 4)

    Returns:
        torch.Tensor: 旋转矩阵批次, shape (B, 3, 3)
    """
    batch_size = quaternions.shape[0]
    device = quaternions.device
    dtype = quaternions.dtype

    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # 构建旋转矩阵
    R = torch.empty(batch_size, 3, 3, device=device, dtype=dtype)

    R[:, 0, 0] = 1 - 2 * (qy*qy + qz*qz)
    R[:, 0, 1] = 2 * (qx*qy - qw*qz)
    R[:, 0, 2] = 2 * (qx*qz + qw*qy)
    R[:, 1, 0] = 2 * (qx*qy + qw*qz)
    R[:, 1, 1] = 1 - 2 * (qx*qx + qz*qz)
    R[:, 1, 2] = 2 * (qy*qz - qw*qx)
    R[:, 2, 0] = 2 * (qx*qz - qw*qy)
    R[:, 2, 1] = 2 * (qy*qz + qw*qx)
    R[:, 2, 2] = 1 - 2 * (qx*qx + qy*qy)

    return R

def _batch_slerp(quat1, quat2, t):
    """
    批量四元数的球面线性插值

    Args:
        quat1 (torch.Tensor): 四元数批次1, shape (B, 4)
        quat2 (torch.Tensor): 四元数批次2, shape (B, 4)
        t (torch.Tensor): 插值比率, shape (B,)

    Returns:
        torch.Tensor: 插值结果, shape (B, 4)
    """
    # 确保四元数在同一半球
    dot = torch.sum(quat1 * quat2, dim=-1)  # (B,)

    # 找到需要翻转的四元数
    flip_mask = dot < 0
    quat2_flipped = quat2.clone()
    quat2_flipped[flip_mask] = -quat2_flipped[flip_mask]
    dot_flipped = torch.abs(dot)

    # 避免除零
    eps = 1e-8
    dot_clamped = torch.clamp(dot_flipped, -1.0 + eps, 1.0 - eps)

    # 计算角度
    theta = torch.acos(dot_clamped)  # (B,)
    sin_theta = torch.sin(theta)  # (B,)

    # 处理小角度情况
    small_angle_mask = sin_theta.abs() < eps
    large_angle_mask = ~small_angle_mask

    result = torch.empty_like(quat1)

    # 大角度情况：使用球面插值
    if large_angle_mask.any():
        indices_large = torch.where(large_angle_mask)[0]
        theta_large = theta[indices_large]
        sin_theta_large = sin_theta[indices_large]
        t_large = t[indices_large]

        w1 = torch.sin((1 - t_large) * theta_large) / sin_theta_large
        w2 = torch.sin(t_large * theta_large) / sin_theta_large

        result[indices_large] = (w1.unsqueeze(-1) * quat1[indices_large] +
                                w2.unsqueeze(-1) * quat2_flipped[indices_large])

    # 小角度情况：使用线性插值
    if small_angle_mask.any():
        indices_small = torch.where(small_angle_mask)[0]
        t_small = t[indices_small]
        result[indices_small] = (quat1[indices_small] +
                                t_small.unsqueeze(-1) * (quat2_flipped[indices_small] - quat1[indices_small]))
        # 标准化
        result[indices_small] = F.normalize(result[indices_small], p=2, dim=-1)

    return result
