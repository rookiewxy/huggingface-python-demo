import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login


# 使用huggingface_hub登录
# 注意：请确保将'你的token'替换为实际的token
login(token='x x x', add_to_git_credential=True)
# 尝试使用GPU，如果不可用则回退到CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 从预训练模型创建pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 生成图像
prompt = "a photo of a cat holding a sign that says hello world"
negative_prompt = ""

try:
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]

    # 保存图像
    image.save("sd3_hello_world.png")
    print("图像已成功生成并保存为 sd3_hello_world.png")

except Exception as e:
    print(f"生成图像时出错: {str(e)}")

# 清理
del pipe
torch.cuda.empty_cache()