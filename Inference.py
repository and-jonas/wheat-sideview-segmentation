
import os
import re
from ImageSegmentor import Segmentor

# DEMO
abspath = os.path.abspath(__file__)
d_name = os.path.dirname(abspath)
os.chdir(d_name)
# workdir = d_name + "/data/Inference/images"
workdir = '/home/anjonas/public/Public/Jonas/Data/ESWW010/ImagesAngle'

# ALL DATA
dirs = [f for f in os.listdir(workdir) if '2024' in f]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]


def run():
    # dirs_to_process = [workdir]  # must be a list  # DEMO
    dirs_to_process = dirs  # must be a list  # ALL DATA
    # dir_output = "output"  # DEMO
    dir_output = f'{workdir}/Output'  # ALL DATA
    dir_vegetation_model = "segveg_ff.pt"
    # dir_sideview_model = "sideview_best-epoch=74-step=1875.00.ckpt"
    # dir_sideview_model = "sideview_best-epoch=06-step=805.00.ckpt"
    # dir_sideview_model = "sideview_best-epoch=24-step=700.00.ckpt"
    # dir_sideview_model = "sideview_best-epoch=52-step=1643.00.ckpt"
    # dir_sideview_model = "sideview_ears_best-epoch=37-step=722.00.ckpt"
    dir_sideview_model = "sideview_ff.pt"
    dir_col_model = "segcol_rf.pkl"
    # dir_patch_coordinates = "data/Inference/coordinates"  # DEMO
    dir_patch_coordinates = f'{workdir}/Meta/patch_coordinates'  # ALL DATA
    image_pre_segmentor = Segmentor(dirs_to_process=dirs_to_process,
                                    dir_vegetation_model=dir_vegetation_model,
                                    dir_sideview_model=dir_sideview_model,
                                    dir_col_model=dir_col_model,
                                    dir_patch_coordinates=dir_patch_coordinates,
                                    dir_output=dir_output,
                                    img_type="JPG",
                                    skip_processed=True)
    image_pre_segmentor.process_images()


if __name__ == "__main__":
    run()
