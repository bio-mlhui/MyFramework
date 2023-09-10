import os
from torch.utils.data import Dataset
import json
import re

from .models import (Image, Object, Attribute, Relationship,
                                  Region, Graph, QA, QAObject, Synset)
from tqdm import tqdm
from .utils import parse_region_descriptions, parse_QA, parse_graph
from .local import save_scene_graphs_by_id, parse_graph_local, init_synsets, save_region_graphs_by_id

class VisualGenome(Dataset):
    def __init__(self,
                 dataset_dir='/home/xhh/datasets/read_genome/visual_genome/data',
                 ) -> None:
        
        super().__init__()
        
        with open(os.path.join(dataset_dir, 'image_data.json'), 'r') as f:
            images_dict = json.load(f)  # list[dict]
        
        assert len(images_dict) == 108077
    
        images_dir = os.path.join(dataset_dir, 'images') 
        
        # save scene graph
        scene_graph_dir = os.path.join(dataset_dir, 'by_id')
        
        
        # save_scene_graphs_by_id(data_dir=dataset_dir, image_data_dir=scene_graph_dir)
        # save_region_graphs_by_id(data_dir=dataset_dir, image_data_dir=scene_graph_dir)
        
        self.items = {} # "img_id": {"meta": Image, "regions": list[Region], "qa"}
        
        for data in tqdm(images_dict):
            img_id = data['id'] if 'id' in data else data['image_id']
            
            url = re.match(r'.*/(VG_100K[_2]?.*)', data['url']).group(1)
            url = os.path.join(images_dir, url)
            width = data['width']
            height = data['height']
            coco_id = data['coco_id']
            flickr_id = data['flickr_id']
            self.items[img_id] = {}
            self.items[img_id]['meta'] = Image(img_id, url, width, height, coco_id, flickr_id)
            
              
        with open(os.path.join(dataset_dir, 'region_descriptions.json'), 'r') as f:
            region_descriptions_by_image = json.load(f)
        for image in tqdm(region_descriptions_by_image):
            # list[{"region_id", "image_id", "phrase", "x", "y", "width", "height"}] -> list[Region] 
            self.items[image['id']]["regions"] = parse_region_descriptions(image['regions'], 
                                                                           self.items[image['id']]['meta'])
    
        with open(os.path.join(dataset_dir, 'question_answers.json'), 'r') as f:
            qa_by_images = json.load(f)
        for image in tqdm(qa_by_images):
            # list[{'a_objects', 'question', 'image_id', 'qa_id', 'answer', 'q_objects'}] -> list[QA]
            self.items[image['id']]['qas'] = parse_QA(image['qas'], self.items)

        self.index_to_image_id = {v:k for k,v in zip(self.items.keys(), 
                                                     list(range(len(self.items.keys()))))}
        self.dataset_dir = dataset_dir
    def __getitem__(self, index):
        
        dic = self.items[self.index_to_image_id[index]]
        meta = dic['meta']
        regions = dic['regions']
        qas = dic['qas']
        
        img_id = meta.id
        
        
        # get scene graph
        scene_graph_dir = os.path.join(self.dataset_dir, 'by_id', str(img_id), 'scene_graph.json')
        with open(scene_graph_dir, 'r') as f:
            data = json.load(f)

        scene_graph = parse_graph_local(data, meta)
        scene_graph = init_synsets(scene_graph, os.path.join(self.dataset_dir, 'synsets.json'))
        
        # get region graphs
        region_graphs = []
        for reg in regions:
            reg_id = reg.id
            region_graph_dir = os.path.join(self.dataset_dir, 'by_id', str(img_id), f'region{reg_id}.json')
            with open(region_graph_dir, 'r') as f:
                region_graph = json.load(f)

            region_graphs.append(parse_graph(region_graph, meta))
        
        return meta, regions, qas, scene_graph, region_graphs
    

dataset = VisualGenome()

data = dataset[0]
