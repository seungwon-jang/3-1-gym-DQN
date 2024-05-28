import gymnasium as gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# Gym 환경 설정
env = gym.make("CartPole-v1", render_mode="rgb_array")
state = env.reset()  # 환경 초기화
env.render()  # 환경 시각화

# 학습을 진행할 디바이스 설정
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# DQN 모델 구성
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# namedtuple 객체 생성
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 리플레이 버퍼 클래스
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 입실론 값 설정
eps = 0.8

# 입실론 그리디 정책을 사용하여 액션 선택 함수
def select_action(state):
    sample = random.random()
    if eps <= sample:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# 전처리 여부 확인을 위한 시각화 함수
def plt_preprocessing_image(state, processed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(state)
    plt.subplot(1, 2, 2)
    plt.title("Processed Frame")
    plt.imshow(processed, cmap='gray')
    plt.show()

# 액션 스페이스 개수 확인
n_action_space = env.action_space.n
print(env.action_space)
print(n_action_space)

# 모델 초기화
model = DQN(n_action_space).to(device)
print(model)

# DQN 모델 만들기
policy_net = DQN(n_action_space).to(device)
target_net = DQN(n_action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())

# 리플레이 메모리 및 배치 사이즈 설정
Memory = ReplayMemory(capacity=10000)
batch_size = 100

# 하이퍼파라미터 설정
discount_rate = 0.99
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
loss_fn = nn.MSELoss()

# 모델 최적화 함수
def optimize_model():
    if len(Memory) < batch_size:
        return
    replay_memory = Memory.sample(batch_size)
    state, action, next_state, reward = zip(*replay_memory)
    state = torch.cat(state)
    action = torch.cat(action)
    next_state = torch.cat(next_state)
    reward = torch.tensor(reward, device=device).float()

    policy_Q = torch.gather(policy_net(state), 1, action).squeeze(1)
    with torch.no_grad():
        target_Q = reward + discount_rate * target_net(next_state).max(1)[0]
    loss = loss_fn(policy_Q, target_Q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습 진행
num_epi = 100
time_step = 5000
update_target = 10

for epi in range(num_epi):
    state = env.reset()
    state = state if isinstance(state, np.ndarray) else state['observations']  # 상태를 np.ndarray로 변환
    state = torch.tensor(state, device=device).float().unsqueeze(0)
    Return = 0
    for i in range(time_step):
        env.render()
        new_action = select_action(state)
        observation, reward, done, truncated, info = env.step(new_action.item())
        next_state = observation if isinstance(observation, np.ndarray) else observation['observations']  # 상태를 np.ndarray로 변환
        next_state = torch.tensor(next_state, device=device).float().unsqueeze(0)
        reward = torch.tensor([reward], device=device)

        Memory.push(state, new_action, next_state, reward)

        state = next_state
        optimize_model()
        if epi % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())

        Return += reward.item()
        if done:
            print(f"Episode: {epi}, Return: {Return}")
            if epi == (num_epi - 1):
                plt_preprocessing_image(env.render(), state.squeeze().cpu().numpy())
            break

# 모델 저장하기
torch.save(model.state_dict(), "test_model1.pth")
print("Saved PyTorch Model State to test_model1.pth")
