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

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size= batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N,C,H,W]: {X.shape}")
    print(f"Shape of y: {y.shape}{y.dtype}")
    break

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
    
model = NeuralNetwork().to(device)
print(model)

#손실함수와 옵티마이저 적용
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

#학습시키기
def train(dataloeader, model, loss_fn, optimizer):
    size = len(dataloeader.dataset)
    #dataloeader에서 mini batch 하나씩 가져온다. x는 입력, y는 출력이다.
    for batch, (X,y) in enumerate(dataloeader):
        #계산하는 디바이스로 값을 전달해 준다
        X,y = X.to(device), y.to(device)
        #예측하고, loss_fn 적용
        pred = model(X)
        loss = loss_fn(pred,y)
        #옵티마이저의 그래디언트를 초기화하고, 손실 그래디언트를 계산하고
        optimizer.zero_grad()
        loss.backward()
        #모델의 파라미터를 업데이트 한다.
        optimizer.step()
        #100번째 미니 배치 마다, 손실이 어떻게 되는지 출력한다.
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #모델을 평가모드로 설정하여, 평가하는 도중 그래디언트를 계산하지 않도록 한다.
    model.eval()
    #변수 초기화
    test_loss, correct = 0,0
    #평가 중에 그래디언트를 계산할 필요가 없으니 계산 안함
    with torch.no_grad():
        for X,y in dataloader:
            #지정된 디바이스로 x,y를 이동합니다.
            X,y = X.to(device), y.to(device)
            #예측되는게
            pred = model(X)
            #예측값과 실제 정답 간의 손실을 계산하여, 더한다.
            test_loss += loss_fn(pred, y).item()
            #argmax(1)은 각 행에서 가장 큰 값의 인덱스를 반환한다. 따라서 pred.argmax(1)은 예측된 클래스를 나타낸다.
            #type(torch.float)를 통해 불리안 텐서를 실수 텐서로 변환한다. .sum()은 모든 값을 더하고 .item()은 텐서 값을 스칼라로 변환한다.
            #따라서 미니배치에서 올바르게 분류된 예측의 개수를 계한하여, correct에 더한다.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # 전체 데이터셋의 크기로 나누어서 정확도를 계산한다.
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#epoch 횟수만큼 학습을 진행한다
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
#모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
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
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')