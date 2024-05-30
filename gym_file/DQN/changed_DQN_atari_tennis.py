import gymnasium as gym

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
#전처리한 이미지가 어떤 건지 그래프로 확인하기 위해 추가
import matplotlib.pyplot as plt

import random

#gym 환경 설정
env = gym.make("Assault-v4",render_mode="rgb_array", frameskip=3)

env.reset() #환경 초기화
env.render()  # 환경을 시각화합니다.

#어떤 디바이스에서 학습을 진행할지 결정한다. 만약 가능하면 cuda 셋팅에서 학습을 진행한다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# 전처리 함수
def preprocessing_state(state):
    #흑백이미지로 만들기 (너비, 높이, 채널)의 구성을 띄고 있을 때, 2번인덱스인 (RGB)채널을 기준으로 하여 평균을 내어서, 흑백 이미지를 만든다.
    gray_state = np.mean(state,axis=2).astype(np.uint8)
    #(210, 160) 사이즈의 그림을 90, 80 스케일로 줄인다.
    resize_image = gray_state[10:190:2, ::2]
    #텐서로 변경하기
    resize_image = torch.from_numpy(resize_image).float().unsqueeze(0).unsqueeze(0).to(device)

    return resize_image

#DQN 모델 구성 - 입력으로 90, 80이 들어와야 한다
class DQN(nn.Module):
    def __init__(self,n_action):
        #상속 받은 module의 매서드들 가져오기
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size= 8, stride= 4) #(21, 19)
        self.conv2 = nn.Conv2d(16, 32, kernel_size= 4, stride= 2) #(9, 8)
        self.fc1 = nn.Linear(32*9*8, 256)
        self.fc2 = nn.Linear(256, n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
#name tuple 객채를 생성한다
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
#리플레이 버퍼 클래스
class ReplayMemory(object):

    def __init__(self, capacity):
        # 최대 길이가 capacity인 deque 생성
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#입실론 값 정하기 80% 확률로 넘어간다.
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
steps_done = 0

#일단 입실론 그리디 방법에서 입실론이 동일한 상황을 가정하고 먼저 만들어 보자.
#state 이미지를 입력 받고, 액션 인덱스 하나를 출력하는 함수
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# 전처리 여부 확인을 위한 시각화 함수
def plt_preprocessing_image(state, processed):
    # 원본 프레임 시각화
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(state)

    # 전처리된 프레임 시각화
    plt.subplot(1, 2, 2)
    plt.title("Processed Frame")
    plt.imshow(processed, cmap='gray')

    plt.show()

action_space = 0
action_space = env.action_space
#action_space 개수 구하기
n_action_space = env.action_space.n

#DQN 모델 만들기
policy_net = DQN(n_action_space).to(device)
target_net = DQN(n_action_space).to(device)
#policy_net의 가중치 편향 및 매개변수를 포함하는 상태를 딕셔너리 형대로 반환받고, 그 반환 받은 딕셔너리 형태를 target_network가 업데이트 한다.
target_net.load_state_dict(policy_net.state_dict())
#리플레이 메모리 만들기, 배치 사이즈 설정
Memory = ReplayMemory(capacity= 1000000)
batch_size = 128

#필요한 변수들
discount_rate = 0.99        #할인률 설정
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
loss_fn = nn.MSELoss()

# 모델 최적화 하는 부분
def optimize_model():
    if len(Memory) < batch_size:
        return
    replay_Memory = Memory.sample(batch_size)
    
    state_batch = torch.cat([transition.state for transition in replay_Memory])
    action_batch = torch.cat([transition.action for transition in replay_Memory])
    reward_batch = torch.cat([torch.tensor([transition.reward], device=device) for transition in replay_Memory])
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, [transition.next_state for transition in replay_Memory])), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([transition.next_state for transition in replay_Memory if transition.next_state is not None])

    policy_Q = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    target_Q = reward_batch + (discount_rate * next_state_values)
    
    loss = loss_fn(policy_Q, target_Q.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#일단 에피소드 진행하기 관련 변수들
num_eqi = 5            #에피소드 개수
time_step = 5000        #타임스텝
update_target = 10      #타겟 네트워트를 몇번의 학습마다 업데이트 할 건가

for epi in range(num_eqi):
    #환경 초기화
    state, info = env.reset()
    Return = 0
    # 초기 이미지 먼저 전처리
    state = preprocessing_state(state)
    for i in range(time_step):
        # 입실론 그리디 알고리즘에 의해 다음 액션 선택
        new_action = select_action(state)
        # 수행 이후 데이터 받아오기
        new_state, reward, done, truncated, info = env.step(new_action.item())
        # 다음 상태도 전처리 하기
        new_state = preprocessing_state(new_state)
        # 만약 마지막이면, 다음 상태 데이터는 None으로 넣어주기
        if done:
            new_state = None
        #리플레이 버퍼에 넣기
        Memory.push(state, new_action, new_state, reward)
        #다음 상태로 넘어가기 - 전처리된 데이터가 들어간다
        state = new_state
        #모델 최적화
        optimize_model()

        Return += reward
        if done == True:
            print(f"epi: {epi}, total_time_step: {i}, Value: {Return}")
            #마지막 종료 시에는 이미지 그래프 출력
            break 
    #정해진 횟수마다 정책 네트워크 파라미터를 타겟 네트워크 파라미터로 넣기
    if epi % update_target == 0:
        target_net.load_state_dict(policy_net.state_dict())

#모델 저장하기
torch.save(policy_net.state_dict(), "test_model1.pth")
print("Saved PyTorch Model State to model.pth")