# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
import cv2


class CocoDetection(VisionDataset):

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()
        '''
    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')'''
        
    def get_image(self, path, start_idx, end_idx):
        # 비디오 이름과 프레임 인덱스 추출
        video_name = path.split('/')[1]  # 비디오 이름 추출
        video_path = os.path.join(self.root, video_name)  # 비디오 경로 설정
        
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)  # 시작 프레임 설정
            
            for frame_index in range(start_idx, end_idx):
                ret, frame = cap.read()
                if not ret:
                    # 프레임을 읽지 못하면 중지
                    print(f"Warning: Failed to read frame {frame_index} from {video_path}.")
                    break  
                frames.append(Image.fromarray(frame).convert('RGB'))
    
            cap.release()
    
            if not frames:
                print(f"Warning: No frames read from {video_path} between indices {start_idx} and {end_idx}.")
                return None
    
            print(f"읽은 프레임의 수: {len(frames)} (범위: {start_idx}-{end_idx})")
            return frames
    
        except Exception as e:
            print(f"Error: An exception occurred while reading frames from {video_path}: {e}")
            return None
                
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)