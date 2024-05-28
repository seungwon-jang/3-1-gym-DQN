import gym
import numpy as np
import matplotlib.pyplot as plt

def preprocess_frame(frame):
    # 원래 크기: (210, 160, 3)
    # 그레이스케일 변환
    gray_frame = np.mean(frame, axis=2).astype(np.uint8)  # (210, 160)
    
    # 슬라이싱: 1번 행부터 175번 행까지, 모든 열에 대해 2열씩 건너뜀
    cropped_frame = gray_frame[::2, ::2]  # (88, 80)
    
    # 정규화: [0, 1] 범위로 변환
    normalized_frame = cropped_frame / 255.0
    
    return normalized_frame

# Atari 환경 생성
env = gym.make('Pong-v0', render_mode= 'rgb_array')
state = env.reset()

# 초기 프레임 렌더링
frame = env.render()

# 전처리된 프레임
processed_frame = preprocess_frame(frame)

# 원본 프레임 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Frame")
plt.imshow(frame)

# 전처리된 프레임 시각화
plt.subplot(1, 2, 2)
plt.title("Processed Frame")
plt.imshow(processed_frame, cmap='gray')

plt.show()

env.close()
