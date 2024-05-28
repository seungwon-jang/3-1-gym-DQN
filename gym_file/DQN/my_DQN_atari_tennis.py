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
env = gym.make("Assault-v4",render_mode="rgb_array")

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
eps = 0.8

#일단 입실론 그리디 방법에서 입실론이 동일한 상황을 가정하고 먼저 만들어 보자.
#state 이미지를 입력 받고, 액션 인덱스 하나를 출력하는 함수
def select_action(state):
    #무작위 숫자 만들기
    sample = random.random()
    #입실론 확률만큼은 최선의 선택하기
    #타임스텝마다 이걸 증가시기게 만드는 수식이 필요할듯
    current_eps = eps
    
    if sample <= current_eps:
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
print(action_space)
print(n_action_space)
#모델을 cpu나 gpu에 올리고, 모델 형태 출력하기
model = DQN(n_action_space).to(device)
print(model)

#DQN 모델 만들기
policy_net = DQN(n_action_space).to(device)
target_net = DQN(n_action_space).to(device)
#policy_net의 가중치 편향 및 매개변수를 포함하는 상태를 딕셔너리 형대로 반환받고, 그 반환 받은 딕셔너리 형태를 target_network가 업데이트 한다.
target_net.load_state_dict(policy_net.state_dict())
#리플레이 메모리 만들기, 배치 사이즈 설정
Memory = ReplayMemory(capacity= 10000)
batch_size = 100 

#필요한 변수들
discount_rate = 0.99        #할인률 설정
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
loss_fn = nn.MSELoss()

# 모델 최적화 하는 부분
def optimize_model():
    #만약 리플레이 버퍼에 저장된 데이터의 크기가 배치 사이즈보다 작으면, -1 반환하면서 아무 것도 하지 않는다.
    if Memory.__len__() < batch_size:
        return -1
    #배치 사이즈 개수 만큼 샘플 가져오기
    replay_Memory = Memory.sample(batch_size)
    #zip은 여러 개의 데이터를 튜플로 만들어주는 연산자 *는 언패킹 연산자이다. 따라서 zip(*뭐시기는) 튜플을 풀어 줘서 맞는 애들끼리 모아준다.
    state, action, next_state, reward = zip(*replay_Memory)
    #텐서들을 연결하는 함수 cat
    state = torch.cat(state)
    action = torch.cat(action)
    next_state = torch.cat(next_state)
    reward = torch.tensor(reward)

    #정책에 의해 선택된 action 값에 따라 나온 Q 값
    policy_Q = torch.gather(policy_net(state), 1, action).squeeze(1)
    #액션을 취한 상태에서 그 자리의 Q 값을 계산 
    with torch.no_grad():
        target_Q =  reward + discount_rate * target_net(next_state).max(1)[0]
    loss = loss_fn(target_Q,policy_Q)

    #네트워크 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#일단 에피소드 진행하기 관련 변수들
num_eqi = 200            #에피소드 개수
time_step = 5000        #타임스텝
update_target = 10      #타겟 네트워트를 몇번의 학습마다 업데이트 할 건가

for epi in range(num_eqi):
    #환경 초기화
    env.reset()
    Return = 0
    for i in range(time_step):
        # 환경을 시각화
        game_image = env.render()
        # 이미지 전처리
        state = preprocessing_state(game_image)
        # 입실론 그리디 알고리즘에 의해 다음 액션 선택
        new_action = select_action(state)
        # 수행 이후 데이터 받아오기
        new_state, reward, done, truncated, info = env.step(new_action.item())
        # 다음 상태도 전처리 하기
        new_state = preprocessing_state(new_state)
        #리플레이 버퍼에 넣기
        Memory.push(state, new_action, new_state, reward)
        #모델 최적화
        optimize_model()
        #정해진 횟수마다 정책 네트워크 파라미터를 타겟 네트워크 파라미터로 넣기
        if epi % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())

        Return += reward
        if done == True:
            print(f"epi: {i}, Value: {Return}")
            #마지막 종료 시에는 이미지 그래프 출력
            if epi == (num_eqi - 1):
                #텐서에서 크기가 1인 차원 삭제
                state = state.squeeze()
                plt_preprocessing_image(game_image,state)
            break 
    if Return > 600:
        eps = 0.85
    elif Return > 1000:
        print("wanted_score")
        break

#모델 저장하기
torch.save(model.state_dict(), "test_model1.pth")
print("Saved PyTorch Model State to model.pth")