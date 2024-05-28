import gymnasium as gym

# Atari 게임 환경을 불러옵니다.
env = gym.make('ALE/Assault-v5', render_mode = 'rgb_array')

Return = 0

timestep = 500  
eoisodes = 1
env.reset()
env.render()  # 환경을 시각화합니다.

env = gym.wrappers.RecordVideo(env, "recording",episode_trigger=None,
                               step_trigger= None,
                               video_length=0, name_prefix= 'rl_video',
                               disable_logger= False)

# 게임을 실행하고 결과를 확인합니다.
for i in range(eoisodes):
    Return = 0
    #환경을 리셋한다
    env.reset()
    for _ in range(timestep):
        env.render()  # 환경을 시각화합니다.
        action = env.action_space.sample()  # 무작위 액션을 선택합니다.
        new_state, reward, done, truncated, info = env.step(action)  # 액션을 적용하고 결과를 받습니다.

        Return = Return + reward
        if done:  # 게임이 종료되면 루프를 중단합니다.
            break

    print("'eposodes: {}, Return: {}".format(i,Return))

# 환경을 닫습니다.
env.close()
