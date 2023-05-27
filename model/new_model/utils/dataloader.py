from torch.utils.data import Dataset


class Fake_Dataset(Dataset):
    """
    dataset类
    """
    def __init__(self, x_text, x_idx, y_data):
        '''
        init: bool  if True, build dataset by embedding layer, otherwise, read it
        '''
        super().__init__()
        self.clean_text = x_text
        self.x_idx = x_idx
        self.y_data = y_data


    def __getitem__(self, index: int):
        result_dict = {
            'origin': {
                'text': self.clean_text[index],
                'idx': self.x_idx[index],
                },
            'y_data': self.y_data[index]
        }
        return result_dict
    
    def __len__(self):
        return len(self.clean_text)
