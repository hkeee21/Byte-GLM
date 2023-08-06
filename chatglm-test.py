from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGenerationByte
from model.baseline_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig
import argparse
import yaml
import os
import torch


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def load_parameter(model_name: str, engine_use: bool):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
    model = model.eval()

    configuration = ChatGLMConfig(
        bos_token_id=130004, 
        eos_token_id=130005, 
        mask_token_id=130000, 
        gmask_token_id=130001,
        pad_token_id=3,
        use_cache=True,
        vocab_size=130528,
        model_type="chatglm",
        torch_dtype="float16",
        # switch on the accelerating engine
        # engine_use=args.engine_use,
        # tiny=tiny_bool
    )

    if engine_use:
        configuration.engine_use = True
        new_model = ChatGLMForConditionalGenerationByte(configuration)
    else:
        new_model = ChatGLMForConditionalGeneration(configuration)
    
    new_model.load_state_dict(model.state_dict(), strict=True)
    if engine_use:
        for i in range(configuration.num_layers):
            new_model.transformer.layers[i].attention.query_key_value.weight.data = new_model.transformer.layers[i].attention.query_key_value.weight.data.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].attention.dense.weight.data = new_model.transformer.layers[i].attention.dense.weight.data.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].mlp.dense_h_to_4h.weight.data = new_model.transformer.layers[i].mlp.dense_h_to_4h.weight.data.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].mlp.dense_4h_to_h.weight.data = new_model.transformer.layers[i].mlp.dense_4h_to_h.weight.data.transpose(0, 1).contiguous()

    return new_model


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=-1)
    parser.add_argument('--test-case', type=int, default=-1)
    parser.add_argument('--engine-use', default=False, action='store_true')
    args = parser.parse_args()

    torch.ops.load_library('./lib/libths_bytetransformer.so')

    assert args.seq_len > -1 or args.test_case > -1, \
        "either seq len or test case should be assigned !"

    string = " "
    if args.seq_len == 8:
        file_name = os.path.join('./case', '8.yaml')
        case_id = 0
    elif args.seq_len == 16:
        file_name = os.path.join('./case', '16.yaml')
        case_id = 0
    elif args.seq_len == 128:
        file_name = os.path.join('./case', '128.yaml')
        case_id = 0
    elif args.seq_len == 256:
        file_name = os.path.join('./case', '256.yaml')
        case_id = 0
    elif args.seq_len == 512:
        file_name = os.path.join('./case', '512.yaml')
        case_id = args.test_case
    elif args.seq_len == 1024:
        file_name = os.path.join('./case', '1024.yaml')
        case_id = args.test_case
    else:
        dir = './case'
        file_list = os.listdir(dir)
        print(file_list)
        file_name = os.path.join('./case', file_list[args.test_case])
        case_id = 0
    f = open(file_name, 'r')
    file = yaml.load(f, Loader=yaml.FullLoader)
    string = file[case_id]
    model_name = "THUDM/chatglm-6b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_parameter(model_name, args.seq_len)
    model_1 = model.eval()
    model_1.half().to("cuda:0")
    response, history = model.chat(tokenizer, parse_text(string), history=[])
    print(response)