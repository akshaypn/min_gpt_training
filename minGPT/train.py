from mingpt.model import GPT
from mingpt.trainer import Trainer
import torch
from torch.utils.data import Dataset
import pickle


class SortDataset(Dataset):
    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5 and inp.unique().nelement() > self.length // 2:
                continue
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train'
            if inp_split == self.split:
                break
        sol = torch.sort(inp)[0]
        cat = torch.cat((inp, sol), dim=0)
        x = cat[:-1].clone()
        y = cat[1:].clone()
        y[:self.length-1] = -1
        return x, y

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 3
model_config.block_size = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(model_config).to(device)
train_dataset = SortDataset('train')

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()

# Save the model
model_path = "sort_model.pth"
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)

# Inference function
def infer(model, input_sequence, device):
    model.eval()
    input_tensor = torch.tensor(input_sequence, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze(0).argmax(dim=-1).tolist()

# Example of inference
test_input = [2, 0, 1, 1, 0, 2]
print("Input sequence:", test_input)
print("Predicted sorted sequence:", infer(model, test_input, device))
