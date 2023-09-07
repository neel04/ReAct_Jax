# Add imports for all the task files
# Make sure each file has a `MyDataset`
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

import reverse_string

def get_datasets() -> List[Dataset]:
    # Add any new datasets here to the list
    return [
        reverse_string.RevDataset(8, 4),
    ]

def get_dataloaders(bsz: int) -> Dict[str, DataLoader]:
    ''''
    Returns a dictionary of dataloaders for each dataset
    With the key being the name of the dataset, as the class name
    '''
    make_dataloader = lambda dataset: DataLoader(dataset, batch_size=bsz, shuffle=True)
    dataloaders = [make_dataloader(dataset) for dataset in get_datasets()]

    return {dataset.__class__.__name__: dataloader for dataset, dataloader in zip(get_datasets(), dataloaders)}

if __name__ == '__main__':
    all_dataloaders = get_dataloaders(bsz=2)
    dataloader = all_dataloaders['RevDataset']
    dataset = dataloader.dataset # Make explcit dataset object

    # So that attributes can be updated
    dataset.seqlen = 32 #type: ignore
    dataset.length = 16 #type: ignore
    
    for batch in dataset:
        print(batch)
        break