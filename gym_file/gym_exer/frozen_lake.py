import gym
#커스텀 맵은 이렇게 설정한다.
my_map = ["SFGF", "FFFF", "FFFF", "FFFF"]
#환경 불러오기
env = gym.make("FrozenLake-v1", desc = my_map, render_mode="rgb_array")
#환경 리셋하기
env.reset()
#동작을 수행할 타임스템
time_step = 50
print('Time Step 0 :')
#랜더링 하기
env.render()
# 타입스텝 한번 하기
for t in range(time_step):
    random_action = env.action_space.sample()

    new_state, reward, done, truncated, info = env.step(random_action)
    print(env.step(random_action))
    print('Time Step {} :'.format(t+1))

    env.render()

    if done:
        break



