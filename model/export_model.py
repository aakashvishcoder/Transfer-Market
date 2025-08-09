import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size,50)
        self.l2 = nn.Linear(50,25)
        self.l3 = nn.Linear(25,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)

model = NeuralNetwork(8)
model.load_state_dict(torch.load("model_state.pth"))
model.eval()

dummy_input = torch.randn(1,8)

torch.onnx.export(
    model, 
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}},
    opset_version=11
)
