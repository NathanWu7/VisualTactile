import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络A和B
class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        self.fc = nn.Linear(3, 4)  # 示例的线性层，输入维度为5，输出维度为3

    def forward(self, x):
        return self.fc(x)

class NetworkB(nn.Module):
    def __init__(self):
        super(NetworkB, self).__init__()
        self.fc = nn.Linear(6, 1)  # 示例的线性层，输入维度为6，输出维度为1

    def forward(self, x):
        return self.fc(x)

# 创建网络实例和损失函数
network_A = NetworkA()
network_B = NetworkB()

criterion_B = nn.MSELoss()
optimizer_A = optim.Adam(network_A.parameters(), lr=0.001)

# 创建示例数据
input_data = torch.randn(10, 5)  # 10个样本，每个样本有5个特征
target = torch.randn(10, 1)  # 示例目标值

# 设置 part_A_size
part_A_size = 3

# 假设你有一个数据加载器 dataloader
num_epochs = 10

for epoch in range(num_epochs):
    # 将input_data的一部分输入到网络A中
    input_A = input_data[:, :part_A_size]
    print("inputA: ", input_A.size())
    output_A = network_A(input_A)
    print("outputA: ", output_A.size())
    # 将output_A与input_data的另一部分拼接，作为网络B的输入
    input_B = torch.cat((output_A, input_data[:, part_A_size:]), dim=1)
    print("input_B: ", input_B.size())
    # 网络B的前向传播
    output_B = network_B(input_B)

    # 计算网络B的损失并更新参数
    loss_B = criterion_B(output_B, target)
    loss_B.backward()
    optimizer_A.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss B: {loss_B.item()}')

