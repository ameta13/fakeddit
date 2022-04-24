import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import default_collate


class FakedditHybridDataset(Dataset):
    """The text + image dataset class"""

    def __init__(self, dataset_file: str, images_dir: str, img_transform=None, num_classes: int = 2):
        """
        Args:
            dataset_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            img_transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.dataset_frame = pd.read_csv(dataset_file, delimiter='\t')
        self.images_dir = images_dir
        self.img_transform = img_transform
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        assert num_classes in (2, 3, 6), f'Wrong num_classes = {num_classes}, expected 2, 3 or 6!'
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            # Get text embedding, tokenize sentence
            sent = self.dataset_frame.loc[idx, 'clean_title']
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=120,  # Pad & truncate all sentences.
                # pad_to_max_length=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            bert_input_id = bert_encoded_dict['input_ids']
            # And its attention mask (simply differentiates padding from non-padding).
            bert_attention_mask = bert_encoded_dict['attention_mask']
            # Get image path
            img_name = self.dataset_frame.loc[idx, 'id'] + '.jpg'
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.dataset_frame.loc[idx, f'{self.num_classes}_way_label']
            if self.img_transform:
                image = self.img_transform(image)
            return {
                'bert_input_id': bert_input_id,
                'bert_attention_mask': bert_attention_mask,
                'image': image,
                'label': label
            }
        except Exception as e:
            # print(f"Corrupted image {img_name}")
            # raise(e)
            return None


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
