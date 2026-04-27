import os
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import argparse


from read_write_model import (
    read_cameras_binary, write_cameras_binary,
    read_images_binary, write_images_binary,
)


MAX_EDGE_PIXELS = 1600




def get_scale_factor(width, height, max_edge):
    max_dim = max(width, height)
    if max_dim <= max_edge:
        return 1.0
    return max_edge / max_dim



def scale_colmap_data(input_colmap_dir, output_colmap_dir, max_edge=MAX_EDGE_PIXELS):
    """
    Resize COLMAP project image metadata, intrinsics, and 2D observations.
    """
    input_colmap_dir = Path(input_colmap_dir)
    output_colmap_dir = Path(output_colmap_dir)


    output_colmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start COLMAP data resize: {input_colmap_dir} -> {output_colmap_dir}")
    print(f"Target max edge length: {max_edge}")

    # 1. Read COLMAP data.
    try:
        cameras = read_cameras_binary(input_colmap_dir / "cameras.bin")
        images = read_images_binary(input_colmap_dir / "images.bin")

        if (input_colmap_dir / "points3D.bin").exists():
            shutil.copy(input_colmap_dir / "points3D.bin", output_colmap_dir / "points3D.bin")
            print("Copied points3D.bin (no modification needed).")

    except FileNotFoundError as e:
        print(
            f"Error: required COLMAP files not found. Please ensure {input_colmap_dir} contains images.bin and cameras.bin."
        )
        raise e

    scaled_cameras = {}

    # 2. Process cameras (intrinsics).
    print("\nProcessing cameras.bin...")
    for camera_id, camera in cameras.items():
        # Compute resize scale.
        scale = get_scale_factor(camera.width, camera.height, max_edge)

        # Modify only when scaling is needed.
        if scale == 1.0:
            scaled_cameras[camera_id] = camera
            continue

        print(f"  Camera ID {camera_id}: original ({camera.width}x{camera.height}) -> scale {scale:.4f}")

        # 2.1. Scale width and height (must be integers).
        new_width = round(camera.width * scale)
        new_height = round(camera.height * scale)

        # 2.2. Scale intrinsic parameters (fx, fy, cx, cy, etc.).
        new_params = list(camera.params)

        # Common COLMAP camera model handling:
        # scale focal lengths and principal point only; keep distortion params unchanged.

        # PINHOLE/OPENCV: [fx, fy, cx, cy, ...]
        if camera.model in ['PINHOLE', 'OPENCV'] and len(new_params) >= 4:
            new_params[0] *= scale  # fx
            new_params[1] *= scale  # fy
            new_params[2] *= scale  # cx
            new_params[3] *= scale  # cy
        # SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL: [f, cx, cy, ...]
        elif camera.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'] and len(new_params) >= 3:
            new_params[0] *= scale  # f
            new_params[1] *= scale  # cx
            new_params[2] *= scale  # cy
        else:
            print(
                f"Warning: unable to automatically handle model {camera.model} for Camera ID {camera_id} ({len(new_params)} params). Please verify manually."
            )

        # Create a new Camera object.
        scaled_cameras[camera_id] = type(camera)(
            id=camera.id,
            model=camera.model,
            width=new_width,
            height=new_height,
            params=np.array(new_params, dtype=np.float64)
        )

    # Write new cameras.bin.
    write_cameras_binary(scaled_cameras, output_colmap_dir / "cameras.bin")
    print("\nWrote new cameras.bin.")

    scaled_images = {}

    # 3. Process images (2D observations).
    print("Processing images.bin...")
    for image_id, image in images.items():
        camera_obj = cameras[image.camera_id]
        scale = get_scale_factor(camera_obj.width, camera_obj.height, max_edge)

        if scale == 1.0:
            scaled_images[image_id] = image
            continue

        print(f"  Image ID {image_id} ({image.name}): scaling 2D observations...")


        new_xys = image.xys * scale


        scaled_images[image_id] = type(image)(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=new_xys,
            point3D_ids=image.point3D_ids
        )


    write_images_binary(scaled_images, output_colmap_dir / "images.bin")
    print("Wrote new images.bin.")
    return scaled_cameras



def scale_image_files(image_source_dir, image_target_dir, input_colmap_dir, colmap_cameras, max_edge=MAX_EDGE_PIXELS):
    """
    Resize actual image files.

    Additional argument:
    - input_colmap_dir: path to the original COLMAP sparse folder (used to read images.bin)
    """
    image_source_dir = Path(image_source_dir)
    image_target_dir = Path(image_target_dir)
    input_colmap_dir = Path(input_colmap_dir)  # Convert to Path object.

    image_target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStart image file resizing: {image_source_dir} -> {image_target_dir}")

    # Build a mapping from filename to camera_id.
    # Use input_colmap_dir to read images.bin.
    try:
        images_data = read_images_binary(input_colmap_dir / "images.bin")
        file_to_camera_id = {img.name: img.camera_id for img in images_data.values()}
    except FileNotFoundError:
        print(f"Error: unable to read {input_colmap_dir / 'images.bin'}. Image resizing aborted.")
        return

    for filename in os.listdir(image_source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            if filename not in file_to_camera_id:
                # Skip files not listed in images.bin.
                continue

            camera_id = file_to_camera_id[filename]

            # Ensure camera_id exists in scaled cameras.
            if camera_id not in colmap_cameras:
                print(f"Warning: Camera ID {camera_id} not in resized camera list. Skipping {filename}.")
                continue

            camera = colmap_cameras[camera_id]

            # Compute target size.
            # For consistency with cameras.bin, directly use resized camera width/height.
            new_width = camera.width
            new_height = camera.height

            # Check whether resizing is truly needed.

            try:
                img = Image.open(image_source_dir / filename)

                # Compute scale proxy by max edge.
                original_max_dim = max(img.width, img.height)
                target_max_dim = max(new_width, new_height)

                # Resize when target dimensions differ.
                if target_max_dim != original_max_dim:

                    # Use high-quality LANCZOS resize.
                    scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Save resized image.
                    scaled_img.save(image_target_dir / filename)
                    # print(f"  Resized {filename}: original ({img.width}x{img.height}) -> new ({new_width}x{new_height})")

                else:
                    # No resize needed; copy directly.
                    shutil.copy(image_source_dir / filename, image_target_dir / filename)


            except Exception as e:
                print(f"Error: failed to resize image file {filename}: {e}")

    print("Image file resizing completed.")



def main():
    parser = argparse.ArgumentParser(description="Batch resize COLMAP data and images")

    parser.add_argument("--input_colmap", type=str, required=True, help="Original COLMAP sparse folder path")
    parser.add_argument("--input_images", type=str, required=True, help="Original image folder")
    parser.add_argument("--output_colmap", type=str, required=True, help="Output path for resized COLMAP sparse")
    parser.add_argument("--output_images", type=str, required=True, help="Output folder for resized images")
    parser.add_argument("--max_edge", type=int, default=1600, help="Maximum image edge length (default: 1600)")

    args = parser.parse_args()

    os.makedirs(args.output_colmap, exist_ok=True)
    os.makedirs(args.output_images, exist_ok=True)

    print(f"Processing: {args.input_colmap}")
    print(f"Target resolution: Max Edge = {args.max_edge}")
    print(f"Target path = {args.output_colmap}")

    scaled_cameras = scale_colmap_data(
        args.input_colmap,
        args.output_colmap,
        max_edge=args.max_edge
    )

    scale_image_files(
        args.input_images,
        args.output_images,
        args.input_colmap,
        scaled_cameras,
        max_edge=args.max_edge
    )

    print(f"\nDone. Results saved to: {args.output_images}")


if __name__ == "__main__":
    main()