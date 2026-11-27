"""Train the gyroscope PINN and compare its trajectory with SciPy."""

from pathlib import Path

import numpy as np
import torch 

from pinn_model import GyroPINN, time_derivative
from reference_solution import GRAVITY, J_PIVOT, MASS, R_BODY

# 定义反对称矩阵函数，计算角速度的反对称矩阵
def hat(omega):
    """Convert a batch of angular velocities to cross-product matrices."""
    x, y, z = omega.unbind(dim=1) # 沿着列方向拆分为一维张量
    zero = torch.zeros_like(x)
    row_x = torch.stack((zero, -z, y), dim=1)
    row_y = torch.stack((z, zero, -x), dim=1)
    row_z = torch.stack((-y, x, zero), dim=1)
    return torch.stack((row_x, row_y, row_z), dim=1) # 返回一个反对称矩阵


def physics_rates(rotation, omega):
    """Compute the two derivatives required by the equations of motion."""
    inertia = torch.as_tensor(J_PIVOT, dtype=omega.dtype, device=omega.device) # 把三个物理量转成与角速度同数据类型，同设备
    gravity = torch.as_tensor(GRAVITY, dtype=omega.dtype, device=omega.device)
    lever = torch.as_tensor(R_BODY, dtype=omega.dtype, device=omega.device)

    rotation_rate = rotation @ hat(omega) # 计算姿态矩阵的导数
    gravity_force = -MASS * (rotation.transpose(1, 2) @ gravity) # 计算物体坐标系下表达的重力
    torque = torch.linalg.cross(lever.expand_as(gravity_force), gravity_force) # 计算重力力矩
    momentum = omega @ inertia.T # 计算角动量
    gyro_term = torch.linalg.cross(omega, momentum) # 计算w×Jw这一项
    angular_acceleration = torch.linalg.solve(inertia, (torque - gyro_term).T).T # 求解角加速度
    return rotation_rate, angular_acceleration # 返回姿态矩阵导数和角加速度

# 定义物理残差函数
def physics_residuals(model, t):
    rotation, omega = model(t) # N 个时刻的预测形状分别为 (N, 3, 3) 和 (N, 3)。
    rotation_dot = time_derivative(rotation, t) # 求姿态矩阵的时间导数
    omega_dot = time_derivative(omega, t) # 求角速度的时间导数
    required_rotation_dot, required_omega_dot = physics_rates(rotation, omega) # 接收同一批预测状态，调用函数，算出陀螺方程要求的斜率。
    return rotation_dot - required_rotation_dot, omega_dot - required_omega_dot # 返回预测值与方程值误差

# 定义损失函数
def physics_loss(residual_rotation, residual_omega, *, omega_scale=600.0):
    # Scale the residuals before adding losses with different units.
    rotation_loss = (residual_rotation / 10.0).square().mean() # 缩放后取平方，然后对所有时间点和分量取平均
    omega_loss = (residual_omega / omega_scale).square().mean()
    return rotation_loss + omega_loss # 把损失相加


@torch.no_grad() # 函数内部所有代码都不再记录梯度计算图
def compare_with_reference(model): # 定义评估函数
    """拿训练好的模型，和参考解（完全一样的时间点上跑一遍预测，方便和参考解逐点对比。"""
    reference_path = Path(__file__).resolve().with_name("reference.npz")
    with np.load(reference_path) as data:
        times = torch.from_numpy(data["t"]).reshape(-1, 1)
        reference_rotation = torch.from_numpy(data["R"])
        reference_omega = torch.from_numpy(data["omega"])
        reference_position = torch.from_numpy(data["position"])

    if times[0].item() != 0.0 or not np.isclose(
        times[-1].item(), model.duration, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"参考解区间与模型的 [0, {model.duration:g}] 秒不一致，请重新运行 reference_solution.py。"
        )

    rotation, omega = model(times)
    position = rotation @ torch.as_tensor(R_BODY, dtype=rotation.dtype)

    # The trace of the relative rotation equals 1 + 2*cos(angle).
    relative_rotation = reference_rotation.transpose(1, 2) @ rotation
    trace = relative_rotation.diagonal(dim1=1, dim2=2).sum(dim=1)
    angle_error = ((trace - 1) / 2).clamp(-1, 1).acos().rad2deg()
    omega_error = omega - reference_omega
    position_error = (position - reference_position).norm(dim=1)

    print(f"\nSciPy comparison at {len(times)} times")
    print(f"Attitude angle RMSE (deg):       {angle_error.square().mean().sqrt().item():.6f}")
    print(f"Maximum attitude error (deg):    {angle_error.max().item():.6f}")
    print(f"Angular rate component RMSE (rad/s): {omega_error.square().mean().sqrt().item():.6f}")
    print(f"Angular rate x/z RMSE (rad/s):   {omega_error[:, [0, 2]].square().mean().sqrt().item():.6f}")
    print(f"Maximum y-rate error (rad/s):    {omega_error[:, 1].abs().max().item():.9f}")
    print(f"COM distance RMSE (m):           {position_error.square().mean().sqrt().item():.6f}")
    print(f"At t = {times[-1].item():g} s:")
    print("  PINN  omega (rad/s):", omega[-1].tolist())
    print("  SciPy omega (rad/s):", reference_omega[-1].tolist())
    print("  PINN  COM (m):", position[-1].tolist())
    print("  SciPy COM (m):", reference_position[-1].tolist())


# 定义 L-BFGS，从模型的当前权重出发，沿用同一批训练时间点。
def refine_with_lbfgs(
    model, t, max_iters, *, omega_scale=600.0, rounds=1,
    tolerance_grad=1e-10, tolerance_change=1e-13,
):
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0, # 试探步长的基准值
        max_iter=max_iters, # max_iters传值
        tolerance_grad=tolerance_grad, # 梯度停止阈值
        tolerance_change=tolerance_change, # 进展停止阈值
        line_search_fn="strong_wolfe", # 启用强 Wolfe 线搜索
    )

    # 定义计算函数，L-BFGS 试探不同步长时会多次调用 closure，重新计算损失和梯度。
    def closure():
        optimizer.zero_grad() # 清除上一次计算留下的权重梯度
        t.grad = None # 清除时间张量上留下的梯度
        loss = physics_loss(*physics_residuals(model, t), omega_scale=omega_scale)
        loss.backward() # 求损失对网络权重的梯度
        return loss # 返回损失

    # 同一阶段的多轮迭代共用优化器，保留 L-BFGS 的曲率历史。
    for _ in range(rounds):
        optimizer.step(closure)


def train_model():
    torch.manual_seed(0) # 设置随机种子
    model = GyroPINN()
    print("Angular rate parameterization: rotating frame")
    print(f"Y angular rate: fixed at {model.omega_initial[1].item():g} rad/s")
    # 512 个等距配点包含两个端点，共 511 个间隔。0.5 秒内间距约为 0.00097847 秒。
    t = torch.linspace(0.0, model.duration, 512, dtype=torch.float64)
    t = t.reshape(-1, 1).requires_grad_() # 每行一个时间点，并设置为可求导
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 设置优化器，使用Adam算法，更新 model.parameter000s() 里的网络权重和偏置，初始学习率设置为0.001

    updates = 1000
    print(f"Training with Adam ({updates} updates)...")
    for _ in range(updates):
        optimizer.zero_grad() # 清空网络
        t.grad = None  # 清除累积的时间输入梯度。
        residual_rotation, residual_omega = physics_residuals(model, t) # 获取残差
        loss = physics_loss(residual_rotation, residual_omega) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

    # 第一阶段 L-BFGS 沿用 Adam 的残差缩放。
    lbfgs_max_iters = 200 # 一次调用最多做200 次 L-BFGS 迭代
    print(f"\nRefining with L-BFGS (up to {lbfgs_max_iters} iterations)...")
    refine_with_lbfgs(model, t, lbfgs_max_iters)
    # 【半秒精调】从上一阶段的权重继续训练。分母由 600 降到 60，角加速度残差的权重提高 100 倍。
    fine_omega_scale = 60.0
    print("Fine tuning with L-BFGS (2 rounds of up to 300 iterations, omega scale = 60)...")
    refine_with_lbfgs(
        model, t, 300, omega_scale=fine_omega_scale, rounds=2,
        tolerance_grad=1e-11, tolerance_change=1e-15,
    )
    final_loss = physics_loss(*physics_residuals(model, t), omega_scale=fine_omega_scale)
    print("Final scaled loss (omega scale = 60):", final_loss.item())
    return model.eval()


def main():
    model = train_model()
    print("\nAfter Adam + L-BFGS")
    compare_with_reference(model)


if __name__ == "__main__":
    main()
