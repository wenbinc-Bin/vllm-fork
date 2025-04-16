"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""
from argparse import Namespace
from typing import List
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
import os
import time
# os.environ['VLLM_SKIP_WARMUP'] = "True"

# os.environ['VLLM_GRAPH_RESERVED_MEM'] = "0.3"
#os.environ['PT_HPUGRAPH_DISABLE_TENSOR_CACHE'] = "False"

def prompt_qwen2_5_vl_chat(model,question,image_urls: List[str]):
    model_name = model

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    if "omni" in model.lower():
        messages = [{
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                    "Group, capable of perceiving auditory and visual inputs, as well as "
                    "generating text and speech."
                },
            ],
        }, {
            "role": "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]
    else:
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                          tokenize=False,
                                          add_generation_prompt=True)

    stop_token_ids = None
    
    return prompt,stop_token_ids



def test_qwen25_vl_multi(llm, MODEL_PATH,IMAGE_PATH1,IMAGE_PATH2,question):
    
    print("test_qwen25_vl_multi()",IMAGE_PATH1,",",IMAGE_PATH2)
    print(" question=",question)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    
    prompt,stop_token_ids = prompt_qwen2_5_vl_chat(MODEL_PATH,question,[IMAGE_PATH1,IMAGE_PATH2])
    image_inputs1 = Image.open(IMAGE_PATH1).convert("RGB")
    image_inputs2 = Image.open(IMAGE_PATH2).convert("RGB")
    image_inputs = [image_inputs1,image_inputs2]
    

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    # if video_inputs is not None:
    #     mm_data['video'] = video_inputs

    if 'omni' in MODEL_PATH.lower():
        llm_inputs = {
            'prompt': prompt[0],
            'multi_modal_data': mm_data,
        }
    else:
        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }
    sampling_params = SamplingParams(
        temperature=0.0,  max_tokens=128,
        stop_token_ids=stop_token_ids,
    )
    start_time = time.time()
    # iter_time = start_time
    # for i in range(20):
    #     outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    #     tmp_time = time.time()
    #     print(f"{i} iter time {tmp_time-iter_time}s")
    #     iter_time = tmp_time
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    end_time = time.time()
    print(f"total time {end_time-start_time}s")
    generated_text = outputs[0].outputs[0].text


    print(generated_text)  
    print()
    
def test_qwen25_vl_single(llm,MODEL_PATH,IMAGE_PATH,question):
    print()
    print("test_qwen25_vl_single()",IMAGE_PATH)
    print("question=",question)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    
    prompt,stop_token_ids = prompt_qwen2_5_vl_chat(MODEL_PATH,question,[IMAGE_PATH])
    #image_inputs, video_inputs = process_vision_info(messages)
    image_inputs = Image.open(IMAGE_PATH).convert("RGB")
    print("image_inputs=",image_inputs)

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    # if video_inputs is not None:
    #     mm_data['video'] = video_inputs
    if 'omni' in MODEL_PATH.lower():
        llm_inputs = {
            'prompt': prompt[0],
            'multi_modal_data': mm_data,
        }
    else:
        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }
    sampling_params = SamplingParams(
        temperature=0.0,  max_tokens=128,
        stop_token_ids=stop_token_ids,
    )
    start_time = time.time()
    iter_time = start_time
    # for i in range(20):
    #     outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    #     tmp_time = time.time()
    #     print(f"{i} iter time {tmp_time-iter_time}s")
    #     iter_time = tmp_time
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    end_time = time.time()
    print(f"total time {end_time-start_time}s")


    generated_text = outputs[0].outputs[0].text

    print(generated_text)
    print()
  
def test_qwen25_vl_single_bs(llm,MODEL_PATH,IMAGE_PATH_list):
    print()
    print("test_qwen25_vl_single_bs()",IMAGE_PATH_list)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    llm_inputs_list=[]
    question_list = ["Describe this image.","How many persons in the image?","Write a song according to this image."]
    i=0
    for IMAGE_PATH in IMAGE_PATH_list:

        prompt,stop_token_ids = prompt_qwen2_5_vl_chat(MODEL_PATH,question_list[i],IMAGE_PATH)
        
        i=i+1
        #image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = Image.open(IMAGE_PATH).convert("RGB")
        print("image_inputs=",image_inputs)

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }
        
        llm_inputs_list.append(llm_inputs)
        
    sampling_params = SamplingParams(
        temperature=0.0,  max_tokens=128,
        stop_token_ids=stop_token_ids,
    )
    outputs = llm.generate(llm_inputs_list, sampling_params=sampling_params)
    for o in outputs:
        print("--------------------------")
        generated_text = o.outputs[0].text
        print(generated_text)
    print()  
    
def test_qwen25_vl_batch(llm,MODEL_PATH,IMAGE_PATH,IMAGE_PATH1,IMAGE_PATH2):
    print()
    print("test_qwen25_vl_batch():","bs1=",IMAGE_PATH,"bs2=",IMAGE_PATH1,",",IMAGE_PATH2)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=128,
        stop_token_ids=[],
    )
    
    messages1 = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "Describe each image."},
        ],
    }
    ]

    messages2 = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH1},
            {"type": "image", "image": IMAGE_PATH2},
            #{"type": "text", "text": "Identify the similarities between these images."},
            {"type": "text", "text": "How many images are there? Describe each image."},
        ],
    }
    ]
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH,trust_remote_code=True)
    prompt1 = processor.apply_chat_template(
        messages1, tokenize=False, add_generation_prompt=True,
    )
    image_inputs1, video_inputs1 = process_vision_info(messages1)

    mm_data1 = {}
    if image_inputs1 is not None:
        mm_data1['image'] = image_inputs1
    if video_inputs1 is not None:
        mm_data1['video'] = video_inputs1

    llm_inputs1 = {
        'prompt': prompt1,
        'multi_modal_data': mm_data1,
    }

    prompt2 = processor.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True,
    )
    image_inputs2, video_inputs2 = process_vision_info(messages2)

    mm_data2 = {}
    if image_inputs2 is not None:
        mm_data2['image'] = image_inputs2
    if video_inputs2 is not None:
        mm_data2['video'] = video_inputs2

    llm_inputs2 = {
        'prompt': prompt2,
        'multi_modal_data': mm_data2,
    }
    
    # outputs = llm.generate([llm_inputs1,llm_inputs2], sampling_params=sampling_params)
    outputs = llm.generate(llm_inputs2, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    

    print(generated_text)  
    print()
    print(outputs[1].outputs[0].text)
    print()


if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm = LLM(
        model=MODEL_PATH,
        max_num_seqs=64,        
        trust_remote_code=True,
        #block_size=128,
        use_padding_aware_scheduling=True,
        max_model_len=8192,
        #enforce_eager=True,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={'image': 2, 'video': 2},
    )

    # MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
    # llm = LLM(
    #     model=MODEL_PATH,
    #     max_model_len=4096,
    #     max_num_seqs=5,
    #     mm_processor_kwargs={
    #         "min_pixels": 28 * 28,
    #         "max_pixels": 1280 * 28 * 28,
    #         "fps": 1,
    #     },
    #     limit_mm_per_prompt={'image': 2, 'video': 2},
    #     dtype="bfloat16",
    # )
    test_qwen25_vl_single(llm,MODEL_PATH,'duck_1204_784.jpg',"Is there animals in this image?")
    test_qwen25_vl_single(llm,MODEL_PATH,'duck.jpg',"Is there animals in this image?") 
    test_qwen25_vl_single(llm,MODEL_PATH,'demo.jpeg',"What is in this image?")
    test_qwen25_vl_single(llm,MODEL_PATH,'duck.jpg',"How many horse do you see?") 
    test_qwen25_vl_single(llm,MODEL_PATH,'duck.jpg',"Tell me a joke relative to the image?")              

    test_qwen25_vl_single(llm,MODEL_PATH,'cherry_blossom.jpg',"Do you know this building?")

    test_qwen25_vl_single(llm,MODEL_PATH,'Beijing.jpeg',"Where is this scene?") 
        
    test_qwen25_vl_multi(llm,MODEL_PATH,'duck.jpg',"Beijing.jpeg","What is the content of each image?")
    
    test_qwen25_vl_multi(llm,MODEL_PATH,"duck.jpg",'cherry_blossom.jpg',"这两张图片的内容一样吗？分别展现的是什么场景？")
    # test_qwen25_vl_batch(llm ,MODEL_PATH,'duck.jpg',"Beijing.jpeg" ,'cherry_blossom.jpg')
    # test_qwen25_vl_single_bs(llm,MODEL_PATH,['duck.jpg','cherry_blossom.jpg','demo.jpeg'])
    # test_qwen25_vl_single_bs(llm,MODEL_PATH,['duck.jpg'])
    print("after")
    #exit()
    
    # model = args.model_path
    # bs= args.bs
    # run_generate(model, bs)

