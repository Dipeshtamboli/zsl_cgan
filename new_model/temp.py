from resnet_model import resnet18

model = resnet18().cuda()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)