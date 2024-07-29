from MSOEmultiscale import MSOEmultiscale
import torch

# model = MSOEmultiscale()
# states_dict = torch.load(f'/home/akarshkumar0101/nca-alife-data/two_stream_dynamic_model.pth')
# model.load_state_dict(states_dict)
# model = model.eval()


# print number of parameters in model
# num_params = sum(p.numel() for p in model.parameters())
# print(f"# of parameters: {num_params}")

# x1 = torch.randn(4, 1, 32, 32)
# x2 = torch.zeros(4, 1, 32, 32)
# x2[:, :, 5:, 5:] = x1[:, :, :-5, :-5]

# x = torch.stack([x1, x2], dim=-1)
# print(x.shape)
# y = model(x)
# print(y.shape)

# print(y)

# print(y.norm(dim=1).mean())

