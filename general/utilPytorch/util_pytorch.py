import torch.nn.parallel
import torch.utils.data
import torch.nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_pytorch_dataloaders(train_dataroot,val_dataroot,train_dataroot_4imgs,val_dataroot_4imgs,batch_size,img_dim=256):

    transform = transforms.Compose(
     [
         transforms.Resize(img_dim),
         transforms.CenterCrop(img_dim),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),

       # transforms.Normalize([0.5], [0.5]),
     ]
    )
    if train_dataroot != None :
        dataset = datasets.ImageFolder(train_dataroot, transform=transform)
        train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True)
    else:
        train_data_loader = None

    if val_dataroot != None:
        dataset = datasets.ImageFolder(val_dataroot, transform=transform)
        val_data_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=8,drop_last=True)
    else:
        val_data_loader = None
    # dataset to create the images for the thesis
    dataset = datasets.ImageFolder(train_dataroot_4imgs, transform=transform)
    train_data_loader_4imgs = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8,drop_last=True)

    dataset = datasets.ImageFolder(val_dataroot_4imgs, transform=transform)
    val_data_loader_4imgs = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8,drop_last=True)

    return train_data_loader,val_data_loader,train_data_loader_4imgs,val_data_loader_4imgs

def load_model(model, pretrained_path,device ='gpu'):
    weights = torch.load(pretrained_path,map_location=torch.device(device))
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)