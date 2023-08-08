import torch

model = torch.load("Kuzin_M_A_4.6_model2_winner_test_2_10.pt")
model.to(torch.float)
torch.save(model, "./Kuzin_M_A_4.6_model2_winner_test_2_10_float.pt")