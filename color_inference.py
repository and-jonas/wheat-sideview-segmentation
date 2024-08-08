
from ImageSegmentor import ImagePostSegmentor

import os

workdir = '/home/anjonas/public/Public/Jonas/Data/ESWW008/ImagesAngle'

dirs = [f for f in os.listdir(workdir)]
# dirs = [os.path.join(workdir, d) for d in dirs if d  not in ['2022_06_01', '2022_06_23', '2022_05_28']]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]


def run():
    dirs_to_process = dirs
    base_dir = f'{workdir}/Output'
    dir_patch_coordinates = f'{workdir}/Meta/patch_coordinates'
    dir_veg_masks = f'{workdir}/Output/SegVeg/Mask'
    dir_stem_ear_masks = f'{workdir}/Output/SegStemEar/Mask'
    dir_output = f'{workdir}/Output/ColSeg'
    dir_model = 'segcol_rf.pkl'
    image_post_segmentor = ImagePostSegmentor(
        base_dir=base_dir,
        dirs_to_process=dirs_to_process,
        dir_patch_coordinates=dir_patch_coordinates,
        dir_veg_masks=dir_veg_masks,
        dir_stem_ear_masks=dir_stem_ear_masks,
        dir_model=dir_model,
        dir_output=dir_output,
        img_type="JPG",
        mask_type="png",
        overwrite=True,
        save_masked_images=False,
        save_color_masks=True,
        n_cpus=7
    )
    image_post_segmentor.process_images()


if __name__ == "__main__":
    run()
