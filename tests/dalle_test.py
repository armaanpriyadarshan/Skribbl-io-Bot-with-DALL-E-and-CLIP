from min_dalle import MinDalle
import torch

torch.cuda.empty_cache()
model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float16,
    is_mega=False,
    is_reusable=True,
)

print("done")

image = model.generate_image(
    text='desert',
    seed=-1,
    grid_size=1,
    log2_k=6,
    log2_supercondition_factor=5,
    is_verbose=False
)

image.save("desert.png")
