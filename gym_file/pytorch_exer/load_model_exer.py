import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#공개된 MNIST데이터 셋에서 학습 데이터를 내려 받는다
training_data = datasets.FashionMNIST(
    root="data",
    train= True,
    download=True,
    transform=ToTensor(),
)

#테스트 데이터를 내려 받는다.
test_data = datasets.FashionMNIST(
    root="data",
    train= False,
    download=True,
    transform=ToTensor(),
)

#어떤 디바이스에서 학습을 진행할지 결정한다. 만약 가능하면 cuda 셋팅에서 학습을 진행한다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#네트워크를 클래스로 정의한다.
#nn.module의 하위 클래스로 정의한다.
class NeuralNetwork(nn.Module):
    #신경망의 계층들을 초기화 한다.
    def __init__(self):
        #부모 클래스의 매서드를 호출하는 방법 중 한가지 방법
        #부모 클래스의 생성자를 호출하는 것을 의미한다.
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #모델 구성
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    #FORWARD메서드를 통해 입력데이터를 모델에 전달하고 출력을 생성한다.
    def forward(self,x):
        #평탄화
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) #순전파 생성
        return logits

#모델 불러오기
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

#불러온 모델을 사용해서 예측하기
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[10][0], test_data[10][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')