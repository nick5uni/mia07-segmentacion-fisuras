"""PyTorch adapter for Victor to validate in his environment; no model is supplied."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from lector_datos import read_manifest, load_pair


class FisurasDataset(Dataset):
    def __init__(self, root, split='train', size=None):
        self.root = Path(root).resolve()
        self.rows = read_manifest(self.root, split)
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, y = load_pair(self.root, row, self.size)
        return torch.from_numpy(x), torch.from_numpy(y), row['image_id']


def make_loaders(root, batch_size=4, size=None, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(FisurasDataset(root,'train',size),batch_size=batch_size,
                       shuffle=True,generator=generator,num_workers=0,drop_last=False)
    val = DataLoader(FisurasDataset(root,'val',size),batch_size=batch_size,
                     shuffle=False,num_workers=0,drop_last=False)
    return train,val


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    train,val = make_loaders(root)
    x,y,ids = next(iter(train))
    assert x.dtype == y.dtype == torch.float32
    assert list(x.shape) == [4,3,416,416]
    assert list(y.shape) == [4,1,416,416]
    assert set(y.unique().tolist()) == {0.0,1.0}
    print('PyTorch batch check PASS:',tuple(x.shape),tuple(y.shape))
    print('No model was trained. For complete CPU checks run verificar_carga.py.')
