import torch
from transformer.v2 import BigramLanguageModel, encode, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel().to(device=device)
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} \n")

state_dict = torch.load("transformer/model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

prompt = encode("Hello World")
print(decode(model.generate(torch.tensor(
    [prompt], device=device), max_new_tokens=500)[0].tolist()))
