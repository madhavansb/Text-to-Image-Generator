!pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

!pip show torch
# These models are from HUgging face Hub models
model1="dreamlike-art/dreamlike-diffusion-1.0" 
model2="stabilityai/stable-diffusion-xl-base-1.0"

pipe=StableDiffusionPipeline.from_pretrained(model1,torch_dtype=torch.float16,use_safetensors=True)
pipe.to("cuda")

prompt="a single tree with plain lands"

image=pipe(prompt).images[0]

print("prompt : ", prompt)
plt.imshow(image)
plt.axis('off')

filename = "generated_image.png"
image.save(filename)

files.download(filename)
