import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

data = pd.read_csv('cart_pole.csv', thousands=',', header=None)
data = data.drop(0, axis = 1) #
data = data.drop(1, axis = 1)
avr_data = []

def moving_average(avr_range):
    for i in range(len(data)):
        if i < 10:
            avr_data.append(i)
        else:
            avr = 0
            avr = data.iloc[i-avr_range:i]
            avr = avr.mean()
            avr_data.append(avr)
            
moving_average(30)
avr_data = pd.DataFrame(avr_data)
data = pd.concat([data, avr_data])

plt.plot(data)
plt.xlabel('timestep')
plt.ylabel('total_Reward')
plt.show()