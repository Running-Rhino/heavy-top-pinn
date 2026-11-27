"""Predict gyroscope motion using a frame with the fast spin removed."""

import torch
from torch import nn

from reference_solution import OMEGA_0, TIME_END

# 定义将单位四元数转换到旋转矩阵的转换函数    
def quaternion_to_rotation(q):
    # Q = q0 + q1i + q2j + q3k, q0=w，q1=x，q2=y，q3=z，即w为实部，x,y,z为虚部
    w, x, y, z = q.unbind(dim=1) # 按列方向拆分四元数
    row_x = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)), dim=1) # 根据公式求出对应旋转矩阵的第一行
    row_y = torch.stack((2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)), dim=1) # 求出第二行
    row_z = torch.stack((2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), dim=1) # 求出第三行
    return torch.stack((row_x, row_y, row_z), dim=1) #返回一个旋转矩阵3×3

# 定义模型类
class GyroPINN(nn.Module):
    def __init__(self): #定义初始化函数
        super().__init__() # 初始化父类
        self.duration = TIME_END # 求解区间从零开始，长度等于终止时刻。
        self.register_buffer("omega_initial", torch.tensor(OMEGA_0, dtype=torch.float64)) # 把初始角速度注册为缓冲区，与可训练参数分开，并转换为双精度张量，登记到当前模型omgea_inital
        # 【固定 y】三个姿态修正量，加上 x、z 两个角速度修正量。
        self.network = nn.Sequential(
            nn.Linear(1, 32), # 1个输入对应32个神经元，第一层隐藏层
            nn.Tanh(), # 激活函数
            nn.Linear(32, 32), # 第二层隐藏层
            nn.Tanh(), # 激活函数
            nn.Linear(32, 5),
        ).double() 

    def forward(self, t): # 定义前向传播    
        # t 是一个二维张量，形状为 (N, 1)，单位是"秒"
        s = t / self.duration # 将求解区间中的时间缩放到 [0, 1]。
        correction = self.network(s) # 每个时刻返回五个修正量
        # 网络预测的修正，是在中间坐标系下的预测

        # q_raw 四元数，从(1, 0, 0, 0)开始，永远不会出现零范数。
        q_raw = torch.cat((torch.ones_like(s), s * correction[:, :3]), dim=1) # s 的形状为 (N, 1)，每个时刻构造一个修正四元数。
        q = q_raw / torch.linalg.vector_norm(q_raw, dim=1, keepdim=True) # 每一行进行单位化，获得单位四元数q
        rotation_correction = quaternion_to_rotation(q) # 调用函数转换为3×3的姿态修正矩阵
        

        # Include the fast initial spin about the body's y axis explicitly.
        half_angle = 0.5 * self.omega_initial[1] * t # 计算半角，得到绕y轴旋转的基准自旋角
        zero = torch.zeros_like(t) # 创建一个与t形状相同的零张量
        spin_q = torch.cat((torch.cos(half_angle), zero, torch.sin(half_angle), zero), dim=1) # 时刻t的基准自旋四元数，t=0时对应初始姿态
        # 把已有的基准自旋 S 单独存下来，供姿态和角速度共同使用。
        spin_rotation = quaternion_to_rotation(spin_q) # S矩阵就是Ry(150t)，即绕y轴旋转，角度为wt=150t
        rotation = rotation_correction @ spin_rotation # 基准自旋矩阵与姿态修正矩阵相乘，得到最终的旋转矩阵

        # 【固定 y】y 分量的修正量为零，因此始终保持初始值。
        omega_correction = torch.cat((correction[:, 3:4], zero, correction[:, 4:5]), dim=1)
        omega_intermediate = self.omega_initial + s * omega_correction
        # omega = Sᵀu，将中间坐标中的角速度转回物体坐标。
        omega = (spin_rotation.transpose(1, 2) @ omega_intermediate.unsqueeze(-1)).squeeze(-1)
        return rotation, omega # 返回当前时刻的旋转矩阵和角速度的预测


def time_derivative(values, t):
    """Differentiate each output component for a batch of independent times."""
    flat_values = values.reshape(t.shape[0], -1) # 把输入张量values重塑为二维张量，形状为(N, M)，其中N是时间步数，M是输出维度
    columns = [] # 创建一个空列表，用于存储每个输出分量的导数
    for component in flat_values.unbind(dim=1): # 沿列把张量拆开，遍历每个输出分量
        derivative, = torch.autograd.grad(component.sum(), t, create_graph=True) # 对列向量的元素和求导，可以得到各时刻自己的斜率，相当于求偏导，create_graph=True表示创建计算图，以便后续反向传播求梯度
        columns.append(derivative) # 将每个分量的导数添加到列表中
    return torch.cat(columns, dim=1).reshape_as(values) # 将所有导数拼接在一起，并重塑为与输入张量values相同的形状


def main():
    torch.manual_seed(0) # 固定种子
    model = GyroPINN() # 创建网络
    t = torch.tensor(
        [[0.0], [0.0005], [model.duration / 2], [model.duration]],
        dtype=torch.float64, requires_grad=True,
    ) # 检查初态、近初态、区间中点和终点。
    rotation, omega = model(t) # 调用模型，得到当前时刻的旋转矩阵和角速度的预测
    rotation_dot = time_derivative(rotation, t) # 计算旋转矩阵的导数
    omega_dot = time_derivative(omega, t) # 计算角速度的导数
    # 验证初始条件
    identity = torch.eye(3, dtype=t.dtype) # 创建一个3×3的单位矩阵，dtype=t.dtype表示与时间张量t相同的类型
    torch.testing.assert_close(rotation[0], identity, rtol=0, atol=0) # 验证初始时刻的旋转矩阵是否为单位矩阵
    torch.testing.assert_close(omega[0], model.omega_initial, rtol=0, atol=0) # 验证初始时刻的角速度是否为初始角速度
    torch.testing.assert_close(
        rotation.transpose(1, 2) @ rotation,
        identity.expand_as(rotation),
        rtol=0, atol=1e-12, # 验证旋转矩阵是否为正交矩阵
    )
    determinants = torch.linalg.det(rotation) # 计算旋转矩阵的行列式
    torch.testing.assert_close(determinants, torch.ones_like(determinants), rtol=0, atol=1e-12) # 验证行列式是否为1

    # Compare automatic differentiation with a centered difference.
    dt = 1e-6
    with torch.no_grad():
        rotation_plus, omega_plus = model(t + dt)
        rotation_minus, omega_minus = model(t - dt)
        torch.testing.assert_close(rotation_dot, (rotation_plus - rotation_minus) / (2 * dt))
        torch.testing.assert_close(omega_dot, (omega_plus - omega_minus) / (2 * dt))

    print("Checks passed: initial conditions, valid rotations, time derivatives.")
    print(f"Shapes: R {tuple(rotation.shape)}, omega {tuple(omega.shape)}")
    print("Untrained model. The values below only demonstrate the code path.")
    print("At t = 0.0005 s:")
    print("R =\n", rotation[1].detach())
    print("omega =", omega[1].detach())
    print("dR/dt =\n", rotation_dot[1].detach())
    print("domega/dt =", omega_dot[1].detach())


if __name__ == "__main__":
    main()
