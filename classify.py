import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from PIL import Image

PATH = "model.pt"

model = torch.jit.load(PATH)

img = Image.open(sys.argv[1])

all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

input = all_transforms(img)

input = torch.unsqueeze(input, 0)

pred = model(input)

_, predicted = torch.max(pred.data, 1)

value = predicted.item()

if value == 0:
    print("airplane")
elif value == 1:
    print("automobile")
elif value == 2:
    print("bird")
elif value == 3:
    print("cat")
elif value == 4:
    print("deer")
elif value == 5:
    print("dog")
elif value == 6:
    print("frog")
elif value == 7:
    print("horse")
elif value == 8:
    print("ship")
elif value == 9:
    print("truck")
