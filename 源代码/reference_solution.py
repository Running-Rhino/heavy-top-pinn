"""Compute and check a reference trajectory for the fixed-pivot heavy top."""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


TIME_END = 0.5  # SciPy 参考解与 PINN 共用的求解终点，单位为秒。
MASS = 15.0 # 定义变量质量 m=15Kg
R_BODY = np.array([0.0, 1.0, 0.0]) # 定义变量物体坐标系下质心初始坐标
GRAVITY = np.array([0.0, 0.0, 9.81])  # 定义变量重力加速度 实际重力为-mg
J_COM = np.diag([0.234375, 0.46875, 0.234375])  # 定义变量质心惯性张量 假设题目惯性张量的参考点在质心
J_PIVOT = J_COM + MASS * (
    (R_BODY @ R_BODY) * np.eye(3) - np.outer(R_BODY, R_BODY)
) # 定义变量支点惯性张量 计算相对于支点的惯性张量
OMEGA_0 = np.array([0.0, 150.0, -4.61538]) # 定义变量初始角速度

# 定义反对称矩阵函数 计算角速度对应的反对称矩阵
def hat(omega):
    x, y, z = omega
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

# 定义导数计算函数，接收时间t和状态数组
def rhs(t, state):
    """Convert the current state into its time derivative."""
    rotation = state[:9].reshape(3, 3) # 前9个元素组成R
    omega = state[9:] #后三个元素为角速度
    rotation_dot = rotation @ hat(omega) # 计算旋转矩阵导数 R_dot
    torque = np.cross(R_BODY, -MASS * (rotation.T @ GRAVITY)) # 计算重力产生的力矩 r×(-mRTg)
    omega_dot = np.linalg.solve(
        J_PIVOT, torque - np.cross(omega, J_PIVOT @ omega)
    ) # 根据运动方程求解角加速度
    return np.concatenate((rotation_dot.ravel(), omega_dot)) # 将旋转矩阵导数和角速度导数concat，返回长度12的数组，用于后续的积分计算

# 定义积分函数，接收输出结果的时间数组，以及相对和绝对误差
def integrate(times, rtol, atol):
    initial = np.concatenate((np.eye(3).ravel(), OMEGA_0)) # 将单位阵和初始角速度展平为长度12的1维向量
    solution = solve_ivp(
        rhs, (times[0], times[-1]), initial, method="DOP853",
        t_eval=times, rtol=rtol, atol=atol,
    ) # 调用积分器
    if not solution.success:
        raise RuntimeError(solution.message) # 如果积分未成功 拉起错误
    states = solution.y.T #转置
    return states[:, :9].reshape(-1, 3, 3), states[:, 9:] # 分割数组

# 定义画图函数
def save_plot(path, times, position, omega):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7), layout="constrained")
    for index, label in enumerate("xyz"):
        axes[0].plot(times, position[:, index], label=label)
    for index, label in ((0, "x"), (2, "z")):
        axes[1].plot(times, omega[:, index], label=label)
    axes[2].plot(times, omega[:, 1], label="y")
    axes[0].set_ylabel("COM position (m)")
    axes[1].set_ylabel("Body rate (rad/s)")
    axes[2].set_ylabel("Body rate (rad/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_title("Heavy top | SciPy reference")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)

# 定义主函数 组织求解检查保存
def main():
    output_interval = 0.0005
    output_count = round(TIME_END / output_interval) + 1
    # 0.5 秒内输出 1001 个时刻、1000 个间隔。DOP853 的内部积分步长由求解器决定。
    times = np.linspace(0.0, TIME_END, output_count)
    rotation, omega = integrate(times, rtol=1e-10, atol=1e-12)
    rotation_check, omega_check = integrate(times, rtol=1e-12, atol=1e-14)

    position = rotation @ R_BODY # 每个输出时刻对应一个质心位置。
    energy = 0.5 * np.sum(omega * (omega @ J_PIVOT), axis=1) # 计算转动动能
    energy += MASS * (position @ GRAVITY) # 加上重力势能
    orthogonality = np.linalg.norm(
        rotation.transpose(0, 2, 1) @ rotation - np.eye(3), axis=(1, 2) # 求正交误差
    )
    print("Max ||R.T R - I||_F:             ", f"{orthogonality.max():.3e}")
    print("Max |det(R) - 1|:                ", f"{np.max(abs(np.linalg.det(rotation) - 1)):.3e}")
    print("Max absolute energy drift (J):  ", f"{np.max(abs(energy - energy[0])):.3e}")
    print("Max R entry change, tighter tol:", f"{np.max(abs(rotation - rotation_check)):.3e}")
    print("Max omega change (rad/s):       ", f"{np.max(abs(omega - omega_check)):.3e}")
    print("Final COM position (m):         ", position[-1])
    print("Final body angular rate (rad/s):", omega[-1])

    output = Path(__file__).resolve().parent
    np.savez(output / "reference.npz", t=times, R=rotation, omega=omega,
             position=position, energy=energy)
    save_plot(output / "reference.png", times, position, omega)


if __name__ == "__main__":
    main()
