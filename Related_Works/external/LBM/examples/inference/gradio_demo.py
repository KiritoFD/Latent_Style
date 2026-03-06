import glob
import logging
import os
from copy import deepcopy

import gradio as gr
import numpy as np
import PIL
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoModelForImageSegmentation
from utils import extract_object, resize_and_center_crop

from lbm.inference import get_model

PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["GRADIO_TEMP_DIR"] = ".gradio"


if not os.path.exists(os.path.join(PATH, "ckpts", "relighting")):
    logging.info(f"Downloading relighting LBM model from HF hub...")
    model = get_model(
        f"jasperai/LBM_relighting",
        save_dir=os.path.join(PATH, "ckpts", "relighting"),
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
else:
    model_dir = os.path.join(PATH, "ckpts", "relighting")
    logging.info(f"Loading relighting LBM model from local...")
    model = get_model(
        os.path.join(PATH, "ckpts", "relighting"),
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).cuda()
image_size = (1024, 1024)

if not os.path.exists(os.path.join(PATH, "examples")):
    logging.info(f"Downloading backgrounds from HF hub...")
    _ = snapshot_download(
        "jasperai/LBM_relighting",
        repo_type="space",
        allow_patterns="*.jpg",
        local_dir=PATH,
    )


def evaluate(
    fg_image: PIL.Image.Image,
    bg_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):

    ori_h_bg, ori_w_bg = fg_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

    _, fg_mask = extract_object(birefnet, deepcopy(fg_image))

    fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
    fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
    bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])

    img_pasted = Image.composite(fg_image, bg_image, fg_mask)

    img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)

    # paste the output image on the background image
    output_image = Image.composite(output_image, bg_image, fg_mask)

    output_image.resize((ori_h_bg, ori_w_bg))

    return (np.array(img_pasted), np.array(output_image))


with gr.Blocks(title="LBM Object Relighting") as demo:
    gr.Markdown(
        f"""
        # Object Relighting with Latent Bridge Matching
        This is an interactive demo of [LBM: Latent Bridge Matching for Fast Image-to-Image Translation](https://arxiv.org/abs/2503.07535) *by Jasper Research*. This demo is based on the [LBM relighting checkpoint](https://huggingface.co/jasperai/LBM_relighting).
    """
    )
    gr.Markdown(
        """
        If you enjoy the space, please also promote *open-source* by giving a ⭐ to the <a href='https://github.com/gojasper/LBM' target='_blank'>Github Repo</a>.
        """
    )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                fg_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    image_mode="RGB",
                    height=360,
                    # width=360,
                )
                bg_image = gr.Image(
                    type="pil",
                    label="Target Background",
                    image_mode="RGB",
                    height=360,
                    # width=360,
                )

            with gr.Row():
                submit_button = gr.Button("Relight", variant="primary")
            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Number of Inference Steps",
                )

            bg_gallery = gr.Gallery(
                # height=450,
                object_fit="contain",
                label="Background List",
                value=[
                    path
                    for path in glob.glob(
                        os.path.join(PATH, "examples/backgrounds/*.jpg")
                    )
                ],
                columns=5,
                allow_preview=False,
            )

        with gr.Column():
            output_slider = gr.ImageSlider(label="Composite vs LBM", type="numpy")
            output_slider.upload(
                fn=evaluate,
                inputs=[fg_image, bg_image, num_inference_steps],
                outputs=[output_slider],
            )

    submit_button.click(
        evaluate,
        inputs=[fg_image, bg_image, num_inference_steps],
        outputs=[output_slider],
    )

    with gr.Row():
        gr.Examples(
            fn=evaluate,
            examples=[
                [
                    os.path.join(PATH, "examples/foregrounds/2.jpg"),
                    os.path.join(PATH, "examples/backgrounds/14.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/10.jpg"),
                    os.path.join(PATH, "examples/backgrounds/4.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/11.jpg"),
                    os.path.join(PATH, "examples/backgrounds/24.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/19.jpg"),
                    os.path.join(PATH, "examples/backgrounds/3.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/4.jpg"),
                    os.path.join(PATH, "examples/backgrounds/6.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/14.jpg"),
                    os.path.join(PATH, "examples/backgrounds/22.jpg"),
                    1,
                ],
                [
                    os.path.join(PATH, "examples/foregrounds/12.jpg"),
                    os.path.join(PATH, "examples/backgrounds/1.jpg"),
                    1,
                ],
            ],
            inputs=[fg_image, bg_image, num_inference_steps],
            outputs=[output_slider],
            run_on_click=True,
        )

    def bg_gallery_selected(gal, evt: gr.SelectData):
        return gal[evt.index][0]

    bg_gallery.select(bg_gallery_selected, inputs=bg_gallery, outputs=bg_image)

if __name__ == "__main__":

    demo.launch(share=True)
