#내가 만든 DQN으로 카트폴을 학습시켜보자
#이거 됨
import gymnasium as gym

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
#전처리한 이미지가 어떤 건지 그래프로 확인하기 위해 추가
import matplotlib.pyplot as plt

import random

from IPython import display

#gym 환경 설정
env = gym.make("LunarLander-v2",render_mode="rgb_array")

state, info = env.reset() #환경 초기화
env.render()  # 환경을 시각화합니다.

#그래프 설정
# 인터랙티브 모드 활성화
plt.ion()
fig, ax = plt.subplots()

# 시각화 함수
def show_env(env):
    ax.clear()
    ax.imshow(env.render())
    plt.pause(0.001)

#어떤 디바이스에서 학습을 진행할지 결정한다. 만약 가능하면 cuda 셋팅에서 학습을 진행한다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__() #super().__init__()그냥 이거랑 같은 의미
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
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

#입실론 값 정하기 1-eps 확률로 넘어간다.
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


action_space = env.action_space
#action_space 개수 구하기
n_action_space = env.action_space.n
n_observations = len(state)

#DQN 모델 만들기
policy_net = DQN(n_observations,n_action_space).to(device)
target_net = DQN(n_observations,n_action_space).to(device)
policy_net.load_state_dict(torch.load('LunaLander_model1.pth'))
#policy_net의 가중치 편향 및 매개변수를 포함하는 상태를 딕셔너리 형대로 반환받고, 그 반환 받은 딕셔너리 형태를 target_network가 업데이트 한다.
target_net.load_state_dict(policy_net.state_dict())
#리플레이 메모리 만들기, 배치 사이즈 설정
Memory = ReplayMemory(capacity= 20000)
batch_size = 200 

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
num_eqi = 1500            #에피소드 개수
time_step = 2000        #타임스텝 이 예제에서 최대 타임스텝은 1000이다
update_target = 10      #타겟 네트워트를 몇번의 학습마다 업데이트 할 건가
update_pri = 0          #업데이트 최소주기 저장
data_df = pd.DataFrame({'epi' : [], 'end_time_step' : [], 'total_return' : []})            #학습 데이터를 저장할 데이터 프레임
for epi in range(num_eqi):
    #환경 초기화
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    Return = 0
    
    for i in range(time_step):
        # 입실론 그리디 알고리즘에 의해 다음 액션 선택
        new_action = select_action(state)
        # 수행 이후 데이터 받아오기
        new_state, reward, terminate, turncated, info = env.step(new_action.item())
        new_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)

        if terminate or turncated or i == 999:
            done = True
        else:
            done = False
        if done:
            new_state = None
        #reward가 double로 들어와서, 자료형 맞춰주기 위해 float형으로 변경
        reward = float(reward)
        #리플레이 버퍼에 넣기
        Memory.push(state, new_action, new_state, reward)
        #다음 상태로 이동
        state = new_state
        #모델 최적화
        optimize_model()

        Return += reward
        #100번째 에피소드 이후로는 보이기
        if epi > 100:
            show_env(env)

        if done:
            print(f"epi: {epi} last_time_step: {i}, Value: {Return}")
            #에피소드 데이터 프레임에 데이터 저장
            epi_df = pd.DataFrame({'epi' : [epi], 'end_time_step' : [i], 'total_return' : [Return]})
            data_df = pd.concat([data_df, epi_df], ignore_index= True)
            break
    #정해진 횟수마다 정책 네트워크 파라미터를 타겟 네트워크 파라미터로 넣기
    #if epi % update_target == 0:
    #    target_net.load_state_dict(policy_net.state_dict())
    #적어도 0보다 컸던 것을 타겟 네트워크로 사용하여, 학습시켜보자 그러면 착륙하는 것을 위주로 생각할 수도 있을 것 같다.
    update_pri = update_pri + 1
    if Return > 0 and update_pri >= update_target:
        update_pri = 0
        target_net.load_state_dict(policy_net.state_dict())
        print("network is update")

env.close()
#카트폴 시뮬레이션 시각화 할 때
plt.ioff()
plt.show()

#모델 저장하기
torch.save(policy_net.state_dict(), "LunaLander_model1.pth")
print("Saved PyTorch Model State to model.pth")
#기존 데이터들이 저장된 파일에 추가하기
#a 모드가 추가모드, 기존 데이터와 인덱스가 동일하다는 가정 하에 무시하기
data_df.to_csv('LunaLander.csv', mode='a', header=False, index=False)
print("Saved train_data")

#https://github.com/yuchen071/DQN-for-LunarLander-v2
#정석적인 DQN은 아니지만 타겟 네트워크를 높은 점수를 얻었던 모델을 기준으로 학습시켜서, 성능 향상을 이룬 글이 있다