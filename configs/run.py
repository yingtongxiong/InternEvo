import itertools
import os
import subprocess

# 读取模板文件内容
with open('template.py', 'r', encoding='utf-8') as file:
    template = file.read()

# 定义替换变量的值
test1_variables = {
    "num_kv_attention_head": [8, 32],
    # "uly_sp": [1, 2, 4, 8, 16, ],
    "ring_sp": [2, 4, 8, 16, 32, 64],
    "window_size": [1],
    "comm_type": ["p2p_AG"],
    "activation_ckpt": [False, True]
}

variables = test1_variables
root_path = "test1/"


output_folder = root_path
os.makedirs(output_folder, exist_ok=True)

# 生成所有变量组合的笛卡尔积
variable_names = list(variables.keys())
combinations = list(itertools.product(*variables.values()))

# 遍历所有组合并生成文件
for i, combination in enumerate(combinations):
    # 创建一个字典，将变量名称映射到对应的值
    combination_dict = dict(zip(variable_names, combination))
    
    # 根据 ring_sp 计算 uly_sp
    combination_dict['uly_sp'] = 64 // combination_dict['ring_sp']
    
    # 替换模板中的变量
    generated_content = template
    for key, value in combination_dict.items():
        generated_content = generated_content.replace(f"{{{key}}}", str(value))
    
    # 定义输出文件名，可以使用组合索引或其他唯一标识符
    output_filename = f"kv{combination_dict['num_kv_attention_head']}_ring{combination_dict['ring_sp']}_ws{combination_dict['window_size']}_comm_{combination_dict['comm_type']}_ckpt{combination_dict['activation_ckpt']}.py"
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(generated_content)
    
    print(f"文件已生成并保存为 {output_path}")
    
    
    # 生成对应的日志文件名
    log_filename = f"kv{combination_dict['num_kv_attention_head']}_ring{combination_dict['ring_sp']}_ws{combination_dict['window_size']}_comm_{combination_dict['comm_type']}_ckpt{combination_dict['activation_ckpt']}.log"
    log_path = os.path.join(output_folder, log_filename)
    
    # 运行命令
    command = f"srun -p Intern5 -N 8 -n 64 --ntasks-per-node=8 --gpus-per-task=1 python ../train.py --config ./{output_path} --profiling 2>&1 | tee '{log_path}'"
    process = subprocess.run(command, shell=True)
    
    if process.returncode != 0:
        print(f"运行命令时出错：{command}")
    else:
        print(f"命令成功运行：{command}")
    
    