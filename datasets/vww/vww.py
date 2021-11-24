import pyvww
from torchvision import transforms


def vww_get_datasets(load_train=True, load_test=True):

    resolution = (258,320)

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize()
        ])
        
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="./datasets/vww/all", annFile="./datasets/vww/annotations_vww/instances_train.json", transform= train_transform)
        
    else:
        train_transform = None
        
        
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            #transforms.Normalize()
        ])

        test_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="./datasets/vww/all", annFile="./datasets/vww/annotations_vww/instances_val.json", transform=test_transform)
        
    else:
        test_dataset = None
        
    return train_dataset, test_dataset










