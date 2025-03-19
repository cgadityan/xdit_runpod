import os
import io
import torch
import numpy as np
from PIL import Image
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from torchvision import transforms
import argparse

def fit_in_box(img, target_width, target_height, fill_color=(255, 255, 255)):
    """Resize image to fit in target box while preserving aspect ratio"""
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        return Image.new("RGB", (target_width, target_height), fill_color)
    
    scale = min(target_width / float(orig_w), target_height / float(orig_h))
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (target_width, target_height), fill_color)
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    new_img.paste(resized, (offset_x, offset_y))
    
    return new_img

def create_inference_image(garment_path, model_path, mask_path, size=(576, 768)):
    """Creates properly formatted input for FLUX Fill model"""
    # Ensure size is a valid tuple with positive dimensions
    if isinstance(size, (int, float)):
        size = (int(size), int(size))
    elif not isinstance(size, tuple) or len(size) != 2:
        size = (576, 768)  # default size
    
    width, height = int(size[0]), int(size[1])
    
    # Add transform
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    try:
        # Load and verify images
        garment_img = Image.open(garment_path).convert("RGB")
        model_img = Image.open(model_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Apply transforms
        garment_tensor = transform(garment_img)
        model_tensor = transform(model_img)
        mask_tensor = mask_transform(mask_img)
        
        # Ensure mask is binary
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Create concatenated image (garment | model)
        inpaint_image = torch.cat([garment_tensor, model_tensor], dim=2)
        
        # Create mask (zeros for garment | mask for model)
        garment_mask = torch.zeros((1, height, width))
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
        
        # Convert tensors back to PIL images
        inpaint_image_pil = transforms.ToPILImage()(inpaint_image * 0.5 + 0.5)
        mask_image_pil = transforms.ToPILImage()(extended_mask)
        
        return inpaint_image_pil, mask_image_pil
        
    except Exception as e:
        raise Exception(f"Error in create_inference_image: {str(e)}")

def process_virtual_try_on(garment_path, model_path, mask_path, output_path, prompt=None, size=(576, 768), seed=42):
    """Process virtual try-on using FLUX Fill model"""
    try:
        # Ensure size is a valid tuple
        if isinstance(size, (int, float)):
            size = (int(size), int(size))
        elif not isinstance(size, tuple) or len(size) != 2:
            size = (576, 768)  # default size
        
        # Create inference images
        combined_image, mask_image = create_inference_image(
            garment_path, 
            model_path, 
            mask_path, 
            size=size
        )
        
        # Use default prompt if none provided
        if prompt is None:
            prompt = "A photo of a person wearing the garment, detailed texture, high quality"
        
        # Run inference
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        result = pipe(
            height=size[1],
            width=size[0] * 2,  # Double width for side-by-side images
            image=combined_image,
            mask_image=mask_image,
            num_inference_steps=50,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=30,
            prompt=prompt,
        ).images[0]
        
        # Extract the right half (try-on result)
        width = size[0]
        tryon_result = result.crop((width, 0, width * 2, size[1]))
        
        # Save the try-on result
        tryon_result = tryon_result.convert('RGB')  # Convert to RGB to ensure compatibility
        tryon_result.save(output_path, format='PNG')
        
        # Create a comparison panel (optional)
        garment_img = Image.open(garment_path).convert("RGB")
        model_img = Image.open(model_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Create blacked-out model
        model_array = np.array(model_img)
        mask_array = np.array(mask_img)
        mask_3d = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)
        model_array[mask_3d > 0] = 0
        model_with_mask = Image.fromarray(model_array)
        
        # Fit images to panel size
        garment_fitted = fit_in_box(garment_img, size[0], size[1])
        model_fitted = fit_in_box(model_with_mask, size[0], size[1])
        result_fitted = fit_in_box(tryon_result, size[0], size[1])
        
        # Create panel
        panel_width = size[0] * 3  # garment + model + result
        panel_height = size[1]
        panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
        
        # Paste images
        panel.paste(garment_fitted, (0, 0))
        panel.paste(model_fitted, (size[0], 0))
        panel.paste(result_fitted, (size[0] * 2, 0))
        
        # Save panel
        panel_path = os.path.splitext(output_path)[0] + "_panel.png"
        panel.save(panel_path, format='PNG')
        
        print(f"Try-on result saved to: {output_path}")
        print(f"Comparison panel saved to: {panel_path}")
        
        return tryon_result
        
    except Exception as e:
        raise Exception(f"Error in virtual try-on processing: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX Fill Virtual Try-On")
    parser.add_argument("--model", default="data/model.jpg", help="Path to model image")
    parser.add_argument("--garment", default="data/garment.jpg", help="Path to garment image")
    parser.add_argument("--mask", default="data/mask.png", help="Path to mask image")
    parser.add_argument("--output", default="data/output.jpg", help="Path to save output image")
    parser.add_argument("--prompt", default="A photo of a person wearing the garment, detailed texture, high quality", 
                        help="Text prompt for generation")
    parser.add_argument("--size", default="1224,1632", help="Output size as width,height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--cache_dir", default="../hf_cache/hub", help="Cache directory for models")
    
    args = parser.parse_args()
    
    # Parse size
    try:
        width, height = map(int, args.size.split(','))
        size = (width, height)
    except:
        print("Invalid size format. Using default 720x960.")
        size = (720, 960)
    
    # Use default prompt if None was provided (this is redundant with the default parameter but kept for clarity)
    prompt = args.prompt if args.prompt else "A photo of a person wearing the garment, detailed texture, high quality"
    
    print("Loading FLUX Fill model...")
    # Initialize the pipeline with the correct model
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", 
        torch_dtype=torch.bfloat16,
        subfolder="transformer",
        cache_dir=args.cache_dir
    )
    # Access Token
    # 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
    # torchrun --nproc_per_node=4 ./hdit.py --model black-forest-labs/FLUX.1-dev --data_parallel_degree 2 --ulysses_degree 2 --ring_degree 2 --prompt "A snowy mountain" "A small dog" --num_inference_steps 50
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    ).to("cuda")
    
    print("Processing virtual try-on...")
    process_virtual_try_on(
        args.garment,
        args.model,
        args.mask,
        args.output,
        prompt=prompt,
        size=size,
        seed=args.seed
    )
