from ast import Continue
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
from imageio import imread
import numpy as np
import pickle
import os
from lib.blob import prep_im_for_blob, im_list_to_blob
from tqdm import tqdm
import torch
import torchvision
from torch.multiprocessing import Manager


class AG(torchvision.datasets.CocoDetection, Dataset):

    _shared_manager = None
    _shared_cache = None

    @classmethod
    def initialize_shared_cache(cls):
        if cls._shared_manager is None:
            cls._shared_manager = Manager()
            cls._shared_cache = cls._shared_manager.dict()
            

    def __init__(self, mode, datasize, feature_extractor, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False, custom_data=None):

        # 경로 설정
        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')
        self.feature_extractor = feature_extractor

        self.initialize_shared_cache()
        self.cached_data = self._shared_cache

        # 객체 클래스 정의 - 클래스 이름을 더 명확한 이름으로 수정
        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # 관계 클래스 정의 - 클래스 이름을 더 명확한 이름으로 수정
        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        self.rel_categories = self.relationship_classes
        self.relation_classes_len = len(self.rel_categories)

        self.video_size = []
        self.gt_annotations = []

        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(root_path + '/annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open('dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(root_path + 'annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(root_path+'annotations/object_bbox_and_relationship.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        print('--------------------finish!-------------------------')
        datasize = 'mini'
        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:500]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        # 유효한 박스 프레임 찾기
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                # 유효한 프레임을 비디오별로 분류하여 저장
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
            
        # 통계변수 초기화
        self.video_list = [] # 비디오 목록
        self.video_size = [] # (w,h) 비디오 크기
        self.gt_annotations = [] # ground truth annotation
        self.non_gt_human_nums = 0 # 사람 없는 프레임 수
        self.non_heatmap_nums = 0 # 히트맵 없는 프레임 수
        self.non_person_video = 0 # 사람 없는 비디오 ㅅ
        self.one_frame_video = 0 # 단일 프레임 비디오 수
        self.valid_nums = 0 # 총 유효프레임 수

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''

        # 사람 바운딩 박스가 없는 프레임 제거
        # 유효한 프레임 카운트
        
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1

                # 사람과 객체의 바운딩 박스정보 저장
                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                # each frames's objects and human
                for k in object_bbox[j]:

                    # 객체와 사람의 바운딩 박스정보 저장
                    # 각 관계 유형을 텐서로 저장
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)
                

            # 2프레임 이상 비디오만 유효 처리
            # 단일 프레임, 무효 비디오 카운트
            
            if len(video) > 2:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(non_heatmap_nums))
        print('x' * 60)
        

        # 데이터 분할 (train/val) 처리
        if custom_data is None:
            self.split = mode  # 'train' or 'val'
            self.train_data, self.val_data = self.split_data(person_bbox, object_bbox)
            if mode == 'train':
                self.video_list = self.train_data
            else:
                self.video_list = self.val_data
        else:
            # custom_data가 있을 경우 (train_data와 val_data가 이미 분할된 경우)
            self.video_list = custom_data

    # 특정 인덱스의 비디오 프레임 처리
    # 데이터셋에서 index 번째 비디오의 프레임들을 가져와서 모델이 사용할 수 있는 형태로 변환
    
    def __getitem__(self, index):

        print(f"Cached indices: {sorted(self.cached_data.keys())}")
        if index in self.cached_data:
            return self.cached_data[index]
            
        frame_names = self.video_list[index]
        gt_annotations = self.gt_annotations[index]
            
        # 비디오 프레임 이름 가져오기
        processed_ims = []
        im_scales = []
        targets = []  # 모든 프레임의 타겟을 저장할 리스트

        for idx, name in enumerate(frame_names):
            #im = Image.open(os.path.join(self.frames_path, name)) 
            #im = np.array(im)  # numpy 배열로 변환
            im = imread(os.path.join(self.frames_path, name))
            im = im[:, :, ::-1] # rgb -> bgr
            #im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            # gpu 메모리 문제로 이미지 크기 축소
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 300, 500)
            im_scales.append(im_scale)
            processed_ims.append(im)


        gt_annotation_frame = gt_annotations[index]

        relationships_combined = (
                [entry["attention_relationship"] for entry in gt_annotation_frame[1:]] + 
                [entry["spatial_relationship"] for entry in gt_annotation_frame[1:]] + 
                [entry["contacting_relationship"] for entry in gt_annotation_frame[1:]]
            )
        

        targets = []
        
        # subj_idx와 obj_idx 추출
        relationships_combined2 = []
        '''
        for idx, entry in enumerate(gt_annotation_frame[1:], start=1):
            # 기본적으로 첫 번째 객체(index 1)를 주체로, 현재 객체를 목적어로 설정
            
            subj_idx = 1  # 첫 번째 객체
            obj_idx = idx  # 현재 객체
        
            # 각 관계 유형을 새로 결합
            if "attention_relationship" in entry and len(entry["attention_relationship"]) > 0:
                for rel_type in entry["attention_relationship"]:
                    relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
            
            if "spatial_relationship" in entry and len(entry["spatial_relationship"]) > 0:
                for rel_type in entry["spatial_relationship"]:
                    relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
            
            if "contacting_relationship" in entry and len(entry["contacting_relationship"]) > 0:
                for rel_type in entry["contacting_relationship"]:
                    relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
        '''
        for idx, entry in enumerate(gt_annotation_frame[1:], start=1):
            subj_idx = entry["class"]
            for other_entry in gt_annotation_frame[1:]:
                if other_entry != entry:
                    obj_idx = other_entry["class"]
                    
                    # 각 관계 유형을 새로 결합
                    if "attention_relationship" in entry:
                        for rel_type in entry["attention_relationship"]:
                            relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
                    
                    if "spatial_relationship" in entry:
                        for rel_type in entry["spatial_relationship"]:
                            relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
                    
                    if "contacting_relationship" in entry:
                        for rel_type in entry["contacting_relationship"]:
                            relationships_combined2.append([subj_idx, obj_idx, rel_type.item()])
                        
        
        processed_ims = np.array(processed_ims) 
        print("relationships_combined2 : ", relationships_combined2)
        print('x'*60)
     
        # 기존 정보 보존하면서 matcher 형식에 맞게 변환
        target = {
            "image_id": index,
            "class_labels": torch.tensor([entry["class"] - 1 for entry in gt_annotation_frame[1:]], dtype=torch.long),
            "boxes": torch.from_numpy(np.array([entry["bbox"] for entry in gt_annotation_frame[1:]])).float(),
            "orig_size": torch.tensor([processed_ims.shape[1], processed_ims.shape[2]]),
            
            # 기존 추가 정보 보존
            "annotations": {
                "boxes": [entry["bbox"] for entry in gt_annotation_frame[1:]],
                "labels": [entry["class"] for entry in gt_annotation_frame[1:]],
                "relationships": relationships_combined2
            },
            "relationships": relationships_combined2,
            "rel": self._get_rel_tensor(relationships_combined2)
        }
        

    

        pixel_values = torch.tensor(processed_ims).permute(0, 3, 1, 2).float()  # NHWC → NCHW로 변환
        pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]).bool()
        orig_size = torch.tensor([processed_ims.shape[1], processed_ims.shape[2]])  # [H, W]
        size = torch.tensor([pixel_values.shape[2], pixel_values.shape[3]])  # [H, W] after any resizing
        
        # 처리된 데이터 캐시에 저장
        self.cached_data[index] = {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            
            "size": size,
            "labels" : [
                {
                    "class_labels": target["class_labels"][i].unsqueeze(0),  # 개별 요소를 0차원 확장
                    "boxes": target["boxes"][i].unsqueeze(0),  # 개별 요소를 0차원 확장,
                    "rel": self._get_rel_tensor(relationships_combined2),
                    "orig_size": orig_size,
                }
                for i in range(len(target["class_labels"]))
            ],
            "target": target
        }
        torch.cuda.empty_cache()
        del processed_ims
        
        return self.cached_data[index]

    def _get_rel_tensor(self, relationships):
        num_object_queries = 100
        num_rel_types = len(self.relationship_classes)
        rel = torch.zeros([num_object_queries, num_object_queries, num_rel_types])
        
        # 관계 정보를 전치 행렬 형태로 재구성
        if relationships:
            # 각 관계 정보를 분리
            subj_indices = [rel_info[0] for rel_info in relationships]
            obj_indices = [rel_info[1] for rel_info in relationships]
            rel_types = [rel_info[2] for rel_info in relationships]
            
            # 전치 행렬 형태로 변환
            rel_array = np.array([subj_indices, obj_indices, rel_types])
            
            # 텐서 생성
            rel[rel_array[0, :], rel_array[1, :], rel_array[2, :]] = 1.0
        
        return rel
        

    def __len__(self):
        return len(self.video_list)

    def split_data(self, person_bbox, object_bbox):

        all_data = self.video_list
        
        split_index = int(0.8 * len(all_data))  # 80% for training
        random.shuffle(all_data)
        train_data = all_data[:split_index]
        val_data = all_data[split_index:]

        return train_data, val_data

    def getImgIds(self):
        # 각 비디오를 하나의 ID로 처리
        return list(range(len(self.video_list)))

    def getCatIds(self):
        """
        COCO 평가를 위한 카테고리 ID 목록 반환
        object_classes에서 background를 제외한 모든 클래스의 인덱스 반환
        """
        # object_classes는 ['__background__', 'class1', 'class2', ...] 형태
        return list(range(1, len(self.object_classes)))  # background(0)를 제외한 ID 반환

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]


def ag_get_statistics(train_data, must_overlap=True):
    num_classes = len(train_data.object_classes)
    num_predicates = train_data.relation_classes_len

    fg_matrix = np.zeros(
        (num_classes + 1, num_classes + 1, num_predicates),
        dtype=np.int64,
    )

    for idx in tqdm(range(len(train_data.video_list))):
        gt_annotations = train_data.gt_annotations[idx]
        
        for frame_annotation in gt_annotations:
            # 객체 클래스 정보 추출 (person 제외)
            object_classes = [ann['class'] for ann in frame_annotation[1:]]
            
            # 관계 정보 추출
            for obj_ann in frame_annotation[1:]:  # person 제외
                if 'attention_relationship' in obj_ann:
                    for rel in obj_ann['attention_relationship']:
                        fg_matrix[1, obj_ann['class'], rel] += 1  # 1은 person class
                if 'spatial_relationship' in obj_ann:
                    for rel in obj_ann['spatial_relationship']:
                        fg_matrix[1, obj_ann['class'], rel] += 1
                if 'contacting_relationship' in obj_ann:
                    for rel in obj_ann['contacting_relationship']:
                        fg_matrix[1, obj_ann['class'], rel] += 1

    return fg_matrix