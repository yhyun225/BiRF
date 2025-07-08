import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
import pdb

class YePopDataset(Dataset):
    def __init__(
        self,
        cfg,
        return_image_path=False,
        preprocessed_annotation=None,
    ):
        DATASET_POSTFIX = 'metadata/images'
        self.root_path = os.path.join(cfg.root_path, DATASET_POSTFIX)  # '/hdd/.../ye_pop'
        self.dataset = []
        self.return_image_path = return_image_path

        self.is_preprocessed = cfg.is_preprocessed
        self.caption_model_type = cfg.caption_model_type

        self.total = 0

        if preprocessed_annotation is not None:
            self.dataset = torch.load(preprocessed_annotation, weights_only=True)
            self.total = len(self.dataset)
        else:
            chunks = sorted(os.listdir(self.root_path))
            for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc='Loading data chunks...'):
                annotation_path = os.path.join(*[self.root_path, chunk, 'annotation.json'])
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                    self.total += len(annotation)

                    for image_id, info in annotation.items():
                        if self.is_preprocessed:
                            image_path = os.path.join(
                                *[self.root_path, chunk, 'latents', image_id.zfill(9) + '.pt']
                            )
                        else:
                            image_path = os.path.join(
                                *[self.root_path, chunk, 'images', info['filename']]
                            )

                        if not os.path.exists(image_path):
                            continue
                        
                        info['image_id'] = image_id
                        info['image_path'] = image_path

                        if self.is_preprocessed:
                            if self.caption_model_type == 'cogvlm':
                                prompt_embed_path = os.path.join(
                                    *[self.root_path, chunk, 'cogvlm_text_emb', 'prompt_embed', image_id.zfill(9) + '.pt']
                                )
                                pooled_prompt_embed_path = os.path.join(
                                    *[self.root_path, chunk, 'cogvlm_text_emb', 'pooled_prompt_embed', image_id.zfill(9) + '.pt']
                                )
                            elif self.caption_model_type == 'llava':
                                prompt_embed_path = os.path.join(
                                    *[self.root_path, chunk, 'llava_text_emb', 'prompt_embed', image_id.zfill(9) + '.pt']
                                )
                                pooled_prompt_embed_path = os.path.join(
                                    *[self.root_path, chunk, 'llava_text_emb', 'pooled_prompt_embed', image_id.zfill(9) + '.pt']
                                )
                            info['text_embedding'] = {
                                'prompt_embed_path': prompt_embed_path,
                                'pooled_prompt_embed_path': pooled_prompt_embed_path,
                            }
                                
                        self.dataset.append(info)

        self._length = len(self.dataset)

        self.transforms = T.Compose([
            T.Resize(cfg.size),
            T.CenterCrop(cfg.size) if cfg.center_crop else T.RandomCrop(cfg.size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __str__(self):
        lines = []
        lines.append(f'{self.__class__.__name__}')
        lines.append(f'    - Data in annotation: {self.total}')
        lines.append(f'    - Total instances: {self._length}')
        lines.append(f'    - Missing instances: {self.total - self._length}')
        lines.append(f'    - Root path: {self.root_path}')
        lines.append(f'    - Caption model: {self.caption_model_type}')
        lines.append(f'    - Preprocessed data: {self.is_preprocessed}')
        return '\n'.join(lines)
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        data = self.dataset[index]
        
        pixel_values = None
        text_prompt = None

        image_path = data['image_path']

        if self.is_preprocessed:
            # precomputed latents & text embeddings
            pixel_values = torch.load(image_path, weights_only=True)
            prompt_emb = torch.load(data['text_embedding']['prompt_embed_path'], weights_only=True)
            pooled_prompt_emb = torch.load(data['text_embedding']['pooled_prompt_embed_path'], weights_only=True)
            text_prompt = {
                'prompt_embeds': prompt_emb,
                'pooled_prompt_embeds': pooled_prompt_emb
            }
        else:
            # raw images & texts -> latents & text embeddings should be computed during training
            pixel_values = Image.open(image_path).convert('RGB')
            pixel_values = self.transforms(pixel_values)

            if self.caption_model_type == 'cogvlm':
                text_prompt = data['cogvlm_caption']
            elif self.caption_model_type == 'llava':
                text_prompt = data['llava_caption']

        orig_text_prompts = data['alt_txt']
        if self.return_image_path:
            return {
                'pixel_values': pixel_values,
                'text_prompts': text_prompt,
                'orig_text_prompts': orig_text_prompts,
                'image_path': image_path    
            }
        
        return {
            'pixel_values': pixel_values,
            'text_prompts': text_prompt,
            'orig_text_prompts': orig_text_prompts
        }


# DEBUGGING
if __name__ == "__main__":
    import yaml

    with open('configs/train_sd3_loral.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = YePopDataset(cfg)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    data = next(iter(dataloader))