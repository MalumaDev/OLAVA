import json
import sys
import os
from functools import cmp_to_key
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'MMMU', 'eval'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA'))

import random

import numpy as np
from tqdm import tqdm

import hashlib

import PIL
import datasets

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser

import torch
from transformers import BitsAndBytesConfig, pipeline, AutoProcessor, LlavaForConditionalGeneration

from lib.MMMU.eval.utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from lib.MMMU.eval.utils.model_utils import call_llava_engine_df, llava_image_processor
from lib.MMMU.eval.utils.eval_utils import parse_multi_choice_response, parse_open_response

# set USE_FLASH_ATTENTION=1
os.environ["USE_FLASH_ATTENTION"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# sub_ds_list = ['llavar', 'docvqa', 'infographicsvqa', 'chartqa/val', 'scienceqa/val']
# test_file_name = 'converted_output_val.json'
shots = list(range(100))  # frozen set of shots to use, change for selecting specific indices


class CustomLLaVADataset(Dataset):
    def __init__(self, processor, data, skip_image=False):
        self.processor = processor
        self.data = data
        self.skip_image = skip_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.skip_image:
            w_image = ['chart', 'diagram', 'drawing', 'figure', 'graph', 'icon', 'image', 'logo', 'map', 'photo',
                       'picture', 'plot', 'symbol', 'table']
            if not any([w in item["final_input_prompt"] for w in w_image]):
                item["final_input_prompt"] = item["final_input_prompt"].replace("<image>", " ")
        inputs = self.processor(item["final_input_prompt"], images=Image.open(item["image"]), padding=True,
                                return_tensors="pt")
        return inputs | {"id": item["id"]}

    @staticmethod
    def collate_fn(batch):
        result = {}
        for k in batch[0]:
            if k != "id":
                # if "pixel" not in k:
                #     result[k] = pad_sequence([dic[k] for dic in batch], batch_first=True, padding_value=pad_token)
                # else:
                result[k] = torch.concat([dic[k] for dic in batch], dim=0)
            else:
                result[k] = [dic[k] for dic in batch]
        return result


def run_model(args, samples, pipe, device=None, max_new_tokens=20):
    out_samples = dict()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if output_path.exists():
        with open(args.output_path, 'r') as f:
            out_samples = json.load(f)
    samples = [sample for sample in samples if sample['id'] not in out_samples]
    with torch.no_grad():
        for sample in tqdm(samples):
            response = pipe(sample['image'], prompt=sample["final_input_prompt"],
                            generate_kwargs={"max_new_tokens": max_new_tokens})
            response = response[0]["generated_text"].replace(sample["final_input_prompt"].replace("<image>", " "),
                                                             "").strip()
            sample['image'] = None
            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
            save_json(output_path, out_samples)

    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare(bb1, bb2):
    if abs(bb1[0][1] - bb2[0][1]) < 5:
        return bb1[0][0] - bb2[0][0]

    return bb1[0][1] - bb2[0][1]


def conv2sample(args, convs, id, ds, image, _shots=None, load_image=False, ocr=None):
    ret = []
    ocr_text = ""
    # resort ocr, if ocr[i][0][1] - ocr[i-1][0][1] < 5, and ocr[i][0][0] < ocr[i-1][0][0], then swap
    if ocr is not None:
        ocr = sorted(ocr, key=cmp_to_key(compare))

        for i, o in enumerate(ocr):
            if i == 0:
                ocr_text += o[1]
                continue
            if abs(ocr[i - 1][0][1] - o[0][1]) < 5:
                ocr_text += " " + o[1]
            else:
                ocr_text += "\n" + o[1]
    else:
        print(f'No OCR for {id} in {ds}')

    for it in range(0, len(convs), 2):
        cur = {}
        assert convs[it]['from'] == 'human'
        cur['question'] = convs[it]['value'].replace('<image>',
                                                     "USER:<image>" + ocr_text + "\n")  # .replace('<image>\n', '').replace('<image>', '')
        cur['options'] = '[]'
        cur['explanation'] = ''
        if load_image:
            cur['image_1'] = PIL.Image.open(
                os.path.join(args.main_data_dir, image))  # PIL.Image.open()  # args.data_path, ds,
        else:
            cur['image_1'] = os.path.join(args.main_data_dir, image)  # args.data_path, ds,
        for jj in range(2, 8):
            cur[f'image_{jj}'] = None
        cur['img_type'] = f"['{ds}']"
        assert convs[it + 1]['from'] == 'gpt'
        cur['answer'] = convs[it + 1]['value']
        cur['topic_difficulty'] = 'Medium'
        cur['question_type'] = 'short-answer'
        cur['subfield'] = f'{ds}'

        base_id = ds + '_' + str(id)
        for_hash = base_id + ' ' + cur['question'] + ' ' + cur['answer']
        suffix = hashlib.md5(str(for_hash).encode('utf-8')).hexdigest()
        cur['id'] = base_id + '_' + suffix

        if _shots is not None:
            cur['shots'] = _shots

        ret.append(cur)
    return ret


def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_test.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="lib/MMMU/eval/configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="data/processed_data")  # hf dataset path: MMMU/MMMU
    parser.add_argument('--main_data_dir', type=str, default=".")  # hf dataset path: MMMU/MMMU
    parser.add_argument('--model_path', type=str, default="llava-hf/llava-1.5-13b-hf")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--test_file_name', type=str, default='converted_output_test_ocr.json')
    parser.add_argument('--sub_ds_list', type=str,
                        default='docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact,mydoc,myinfographic,mychart')
    parser.add_argument('--eval_llava1_6', action='store_true', help='if llava1.6')
    parser.add_argument('--debug', action='store_true', help='enable debugger')
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--load_weights', type=str, default=None)
    parser.add_argument("--skip_image", type=bool, default=True)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    max_new_tokens = args.max_new_tokens

    if args.debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger('9.67.169.241', 12345)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    # for subject in CAT_SHORT2LONG.values():
    #     sub_dataset = load_dataset(args.data_path, subject, split=args.split)
    #     sub_dataset_list.append(sub_dataset)

    sub_ds_list = args.sub_ds_list.split(',')
    for ds in sub_ds_list:
        if not os.path.exists(os.path.join(args.data_path, ds, args.test_file_name)):
            print(f'File not found found: {os.path.join(args.data_path, ds, args.test_file_name)} -- SKIPPING')
            continue
        with open(os.path.join(args.data_path, ds, args.test_file_name), 'r') as f:
            ds_data = json.load(f)

        # few-shot support
        _shots = None
        if args.shots > 0:
            assert args.shots < len(shots)
            _shots = []
            for ix in range(args.shots):
                c = ds_data[shots[ix]]
                _shots.extend(
                    conv2sample(args, c['conversations'], c['id'], ds, (c['image'] if 'image' in c else 'no_image.jpg'),
                                load_image=False))

        tab_ds = []
        for c in tqdm(ds_data, desc=ds):
            if 'image' not in c:
                continue
            if 'ocr' not in c:
                continue
            tab_ds.extend(conv2sample(args, c['conversations'], c['id'], ds, c['image'], _shots,
                                      ocr=c['ocr']))
        sub_dataset = datasets.Dataset.from_list(tab_ds)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        # if sample['image']:
        #     sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)
    # run ex

    # load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_id = args.model_path

    processor = AutoProcessor.from_pretrained(model_id)

    if args.load_weights:
        model = LlavaForConditionalGeneration.from_pretrained(args.load_weights,
                                                              quantization_config=quantization_config,
                                                              device_map="auto")
    else:
        model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config,
                                                              device_map="auto")

    # if args.load_weights:
    #     tmp = load_file(args.load_weights)
    #     for key in list(tmp.keys()):
    #         tmp[key.replace("model.mm_projector.", "linear_")] = tmp[key]
    #         del tmp[key]
    #     model.load_state_dict(tmp, strict=False)

    out_samples = dict()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if output_path.exists():
        with open(args.output_path, 'r') as f:
            out_samples = json.load(f)
    samples = [sample for sample in samples if sample['id'] not in out_samples]

    for sample in samples:
        if sample['question_type'] == 'multiple-choice':
            raise NotImplementedError("Multiple choice questions are not supported yet.")

    ds = CustomLLaVADataset(processor, samples, skip_image=args.skip_image)
    dataloader = DataLoader(ds, batch_size=1, num_workers=4,
                            collate_fn=CustomLLaVADataset.collate_fn)

    with torch.no_grad():
        for j, sample in enumerate(tqdm(dataloader)):
            response = model.generate(**{
                k: sample[k].to(device) for k in sample if k != "id"
            }, max_new_tokens=max_new_tokens)
            generated_text = processor.batch_decode(response, skip_special_tokens=True)
            # if sample['question_type'] == 'multiple-choice':
            #     pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            # else:  # open question
            #     pred_ans = response
            # out_samples[sample['id']] = pred_ans
            for i, g in enumerate(generated_text):
                out_samples[sample['id'][i]] = g.split("ASSISTANT:")[-1].split("\n")[0].strip()
            save_json(output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()
