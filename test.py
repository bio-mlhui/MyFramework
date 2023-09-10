# from transformers import pipeline
# from detectron2.utils.visualizer import Visualizer
# from detectron2.utils.video_visualizer import VideoVisualizer
# fix_spelling = pipeline("text2text-generation",model="/home/xhh/pt/spelling-correction-english-base")

# print(fix_spelling(["a baby giant panda near a large giant panda"],max_length=2048))



#####################################video clip#############################
from models.videoclip.models.mmfusion import MMPTModel
from models.videoclip.models import MMFusionSeparate
import yaml
import argparse
import torch
import torchvision.io as video_io

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# with open ('./models/videoclip/how2.yaml', 'r') as f:
#     configs = yaml.safe_load(f)
# configs = dict2namespace(configs)

# model = MMFusionSeparate(config=configs,
#                          checkpoint_path=configs.eval.save_path)

# from models.videoclip.processors.models.s3dg import S3D
# video_encoder = S3D('/home/xhh/workspace/fairseq/examples/MMPT/pretrained_models/s3d_dict.npy', 512)
# video_encoder.load_state_dict(
#     torch.load('/home/xhh/workspace/fairseq/examples/MMPT/pretrained_models/s3d_howto100m.pth'))
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     configs.dataset.bert_name, use_fast=False
# )
# from models.videoclip.processors import Aligner
# aligner = Aligner(configs.dataset)

# videoclip = MMPTModel(configs, model, video_encoder)


# videoclip.eval()

# from einops import rearrange
# # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
# fps=30
# video_frames = video_io.read_video(
#                 filename='./dog.mp4', pts_unit='sec', output_format='TCHW')[0].unsqueeze(0).float() / 255.0
# video_frames = rearrange(video_frames, 'b (t fps) c h w -> b t fps h w c',fps=fps)
# caps, cmasks = aligner._build_text_seq(
#     tokenizer("a girl standing with her mother", add_special_tokens=False)["input_ids"]
# )

# caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

# with torch.no_grad():
#     output = videoclip(video_frames, caps, cmasks, return_score=True)
# print(output["score"])  # dot-product


# #################################################clip###############################
# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# import matplotlib.pyplot as plt
# import torch
# from einops import rearrange, repeat
# import torch.nn.functional as F
# nf = 5
# h = 384
# w = 640

# pad_bottom = 10
# pad_right = 20
# font_size = 20
# linespacing = 2
# title = 'this is a test of long long long long long sentence'
# mlm_masked_title = 'this is a test of long long long long long mlm_masekd_title'
# mlm_pred_title = 'this is a test of long long long long long mlm_pred_title'
# font_length = len(mlm_pred_title)

# vid_frames = torch.rand([nf, 3, h, w])# t 3 h w
# vid_frames = F.pad(vid_frames, [0, pad_right, 0, pad_bottom]) # 
# vid_frames = rearrange(vid_frames, 't c h w -> (t h) w c')

# gt_referent = torch.randint(0, 2, [nf, h, w]) # t h w
# gt_referent = F.pad(gt_referent, [0, pad_right, 0, pad_bottom])
# gt_referent = rearrange(gt_referent, 't h w -> (t h) w')
# gt_referent = repeat(gt_referent, 'th w -> th w c',c=3)

# mvm_masked = vid_frames
# mvm_pred = vid_frames

# refer_pred = gt_referent # t h w

# whole_image = torch.hstack([vid_frames, gt_referent, refer_pred, mvm_masked, mvm_pred])
# print(whole_image.shape[1], whole_image.shape[0])

# fig_with = max(whole_image.shape[1], (font_size*font_length))

# fig, axs = plt.subplots(figsize=(fig_with/100, (whole_image.shape[0]+((3+linespacing*2)*font_size))/100,))
# axs.xaxis.set_visible(False)
# axs.yaxis.set_visible(False)
# axs.imshow(whole_image)
# # axs['video'].imshow(vid_frames)
# # axs['refer_pred'].imshow(refer_pred)
# # axs['mvm_masked'].imshow(mvm_masked)
# # axs['mvm_pred'].imshow(mvm_pred)

# fig.suptitle(f'{title}\n{mlm_masked_title}\n{mlm_pred_title}', fontsize=font_size, linespacing=2.)
# fig.savefig(f'./test.png')
# plt.close()

# canvas = Visualizer()


    
# anon_map = {}
# attributes = []
# for src, role, tgt in g.attributes():
#     if constant.type(tgt) in (constant.INTEGER, constant.FLOAT):
#         anon_val = f'number_{len(anon_map)}'
#         anon_map[anon_val] = tgt
#         tgt = anon_val
#     attributes.append((src, role, tgt))
# g2 = penman.Graph(g.instances() + g.edges() + attributes)
# print(penman.encode(g2))
# anon_map

# from transformers import BartForConditionalGeneration
# from models.amr_utils.tokenization_bart import AMRBartTokenizer
# amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
# encoder = amr2text_model.get_encoder()
# amr_tokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')

# linearized_amr = '( <pointer:0> hand~0 :ARG1-of ( <pointer:1> move-01~5 ) :part-of ( <pointer:2> man~3 ) )'

# tokenized_amr, meta_dict = amr_tokenizer.tokenize_amr_with_only_each_token_length(linearized_amr.split())
            
class TrainRandomSampler_ByEpoch:
    def __init__(self, 
                 data_source,
                 seed,
                 ) -> None:
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.seed = seed
        self.epoch = None

    def __iter__(self):
        seed = self.seed + self.epoch
        print(f'generating a new indices permutations for this epoch using seed {seed}')
        n = len(self.data_source)
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

datasource = torch.arange(100)

sampler = TrainRandomSampler_StartEpoch(datasource, seed=1999)
epoch5_top10 = []
for epoch in range(10):
    sampler.set_epoch(epoch)
    for item in sampler:
        if epoch == 5:
            if len(epoch5_top10) < 10:
                epoch5_top10.append(item)
        pass
    
new_epoch_top10 = []       
sampler = TrainRandomSampler_StartEpoch(datasource, seed=1999)
for epoch in range(2, 10):
    sampler.set_epoch(epoch)
    for item in sampler:
        if epoch == 5:
            if len(new_epoch_top10) < 10:
                new_epoch_top10.append(item)
        pass

print(epoch5_top10)
print(new_epoch_top10)