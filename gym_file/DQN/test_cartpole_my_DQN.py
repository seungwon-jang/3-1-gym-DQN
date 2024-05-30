#내가 만든 DQN으로 카트폴을 학습시켜보자

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
env = gym.make("CartPole-v1",render_mode="rgb_array")

state, info = env.reset() #환경 초기화
env.render()  # 환경을 시각화합니다.

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
#이 수치로 steps_done을 eps 비율을 결정한다.
eps_decay = 2000
steps_done = 0

#일단 입실론 그리디 방법에서 입실론이 동일한 상황을 가정하고 먼저 만들어 보자.
#state 이미지를 입력 받고, 액션 인덱스 하나를 출력하는 함수
def select_action(state):
    global steps_done
    #무작위 숫자 만들기
    sample = random.random()
    #입실론 확률만큼은 최선의 선택하기
    #타임스텝마다 이걸 증가시기게 만드는 수식이 필요할듯
    current_eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    if sample > current_eps:
        #그래디언트 계산 안한다.
        with torch.no_grad():
            #모델에 의해서 결정된 action중 가장 값이 높은 것을 반환한다.
            #DQN 네트워크의 출력은 action 개수의 크기를 가지는 벡터를 출력으로 가진다
            #.max(1)은 pytorch에서 1차원 별로 최대 값을 찾는 함수이다. 결과는 (값, 인덱스)의 튜플로 나온다
            #예를 들어 torch.max(1)에 [[1,2,3,], [3,4,5]]가 들어간다면 123중 최대값, 인덱스, 345중 최대값, 인덱스가 결과 값으로 나오게 된다.
            #그래서 우리는 [n_action] 하나이니 (최대 값, 인덱스) 튜플을 얻게 되고, 1번에 해당하는 인텍스 값을 텐서로 리턴하려는 것이다.
            return policy_net(state).max(1)[1].view(1,1)
    else:
        #아닐 경우 텐서형태로 랜덤으로 action_space에서 뽑아서 출력한다
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


action_space = env.action_space
#action_space 개수 구하기
n_action_space = env.action_space.n
n_observations = len(state)

#DQN 모델 만들기
policy_net = DQN(n_observations,n_action_space).to(device)
target_net = DQN(n_observations,n_action_space).to(device)
#policy_net의 가중치 편향 및 매개변수를 포함하는 상태를 딕셔너리 형대로 반환받고, 그 반환 받은 딕셔너리 형태를 target_network가 업데이트 한다.
target_net.load_state_dict(policy_net.state_dict())
#리플레이 메모리 만들기, 배치 사이즈 설정
Memory = ReplayMemory(capacity= 10000)
batch_size = 100 

#필요한 변수들
discount_rate = 0.90        #할인률 설정
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
loss_fn = nn.MSELoss()

# 모델 최적화 하는 부분
def optimize_model():
    #만약 리플레이 버퍼에 저장된 데이터의 크기가 배치 사이즈보다 작으면, -1 반환하면서 아무 것도 하지 않는다.
    if Memory.__len__() < batch_size:
        return 
    #배치 사이즈 개수만큼 샘플 가져오기
    replay_Memory = Memory.sample(batch_size)
    #none인 애들 뺴기 - 이거 안되는 듯 하다 일단 해보자
    replay_Memory = [transition for transition in replay_Memory if transition.next_state is not None]
    
    #zip은 여러 개의 데이터를 튜플로 만들어주는 연산자 *는 언패킹 연산자이다. 따라서 zip(*뭐시기는) 튜플을 풀어 줘서 맞는 애들끼리 모아준다.
    state, action, next_state, reward = zip(*replay_Memory)
    #텐서들을 연결하는 함수 cat
    state = torch.cat(state)
    action = torch.cat(action)
    next_state = torch.cat(next_state)
    reward = torch.tensor(reward, device=device, dtype=torch.bool)

    #정책에 의해 선택된 action 값에 따라 나온 Q 값
    #policy_Q = torch.gather(policy_net(state), 1, action).squeeze(1)
    policy_Q = torch.gather(policy_net(state), 1, action)
    #액션을 취한 상태에서 그 자리의 Q 값을 계산 
    with torch.no_grad():
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values = target_net(next_state).max(1)[0]
        target_Q =  reward + discount_rate * next_state_values
    loss = loss_fn(policy_Q, target_Q.unsqueeze(1))

    #네트워크 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#일단 에피소드 진행하기 관련 변수들
num_eqi = 1000            #에피소드 개수
time_step = 5000        #타임스텝
update_target = 10      #타겟 네트워트를 몇번의 학습마다 업데이트 할 건가

for epi in range(num_eqi):
    #환경 초기화
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    Return = 0
    for i in range(time_step):
        # 입실론 그리디 알고리즘에 의해 다음 액션 선택
        new_action = select_action(state)
        # 수행 이후 데이터 받아오기
        new_state, reward, done, truncated, info = env.step(new_action.item())

        if done == True:
            print(f"epi: {epi} last_time_step: {i}, Value: {Return}")
            break

        new_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        #리플레이 버퍼에 넣기
        Memory.push(state, new_action, new_state, reward)
        
        #다음 상태로 이동
        state = new_state
        #모델 최적화
        optimize_model()
        
        Return += reward
    #정해진 횟수마다 정책 네트워크 파라미터를 타겟 네트워크 파라미터로 넣기
    if epi % update_target == 0:
        target_net.load_state_dict(policy_net.state_dict())

#모델 저장하기
torch.save(policy_net.state_dict(), "test_model1.pth")
print("Saved PyTorch Model State to model.pth")