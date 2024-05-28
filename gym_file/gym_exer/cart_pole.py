import gym

env = gym.make('CartPole-v1', render_mode ="rgb_array")

episodes = 100
timestep = 50

for i in range(episodes):
    #총 이득 계산
    Return = 0
    state = env.reset()
    for t in range(timestep):
        env.render()
        random_action = env.action_space.sample()

        new_state, reward, done, truncated, info = env.step(random_action)

        Return = Return + reward

        if done:
            break
    
    if i%10 == 0:
        print('episode: {}, return: {}'.format(i,Return))

env.close()

        



