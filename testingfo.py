
import fiftyone as fo
from flash.image import SemanticSegmentation, SemanticSegmentationData
from flash.image.segmentation.output import FiftyOneSegmentationLabelsOutput
from flash import Trainer
from transforms2 import set_input_transform_options
from itertools import chain


# define input transform for each task
transform = set_input_transform_options(train_size=600,
                                        crop_factor=0.737,
                                        p_color_jitter=0,
                                        blur_kernel_size=1,
                                        predict_scale=1,
                                        )

# dataset_train = fo.Dataset.from_dir(
#     dataset_dir="C:/Users/anjonas/PycharmProjects/WheatSideviewSegmentation/data/validation",
#     dataset_type=fo.types.ImageSegmentationDirectory,
#     # max_samples=10,
#     data_path="./images",
#     labels_path="./masks",
#     force_grayscale=False,
#     shuffle=True,
#     tags=["train"]
# )
#
# dataset_validation = fo.Dataset.from_dir(
#     dataset_dir="C:/Users/anjonas/PycharmProjects/WheatSideviewSegmentation/data_stems/validation",
#     dataset_type=fo.types.ImageSegmentationDirectory,
#     max_samples=2,
#     data_path="./images",
#     labels_path="./masks",
#     force_grayscale=False,
#     shuffle=True,
#     tags=["validation"]
# )
#
# session = fo.launch_app(dataset_train)
# session.wait()
# session.close()

# 4. Segment a few images!
# TODO loading from checkpoint (.ckpt) results in CUDA out of memory
model = SemanticSegmentation.load_from_checkpoint(
    "/projects/segment-sideview/sideview_final-epoch=70-step=1917.00.ckpt",
)

dataset = fo.Dataset.from_dir(
    # dataset_dir="/projects/SegVeg2/data/validation/images",
    # dataset_dir="/projects/SegVeg2/data/prediction",
    dataset_dir="/projects/segment-sideview/data/validation/images",
    dataset_type=fo.types.ImageSegmentationDirectory,
    max_samples=1,
    data_path=".",
    labels_path=".",
    force_grayscale=True,
    shuffle=True,
)

# datamodule = SemanticSegmentationData.from_fiftyone(
#     predict_dataset=predict_dataset,
#     batch_size=5,
#     transform_kwargs=dict(image_size=(592, 592)),
#     num_classes=2,
#     num_workers=8,
#     predict_transform=SemSegInputTransform,
#     pin_memory=True,
#     persistent_workers=True
# )
datamodule = SemanticSegmentationData.from_fiftyone(
    predict_dataset=dataset,
    batch_size=1,
    predict_transform=transform,
    num_classes=3,
    num_workers=8,
    # pin_memory=True,
    # persistent_workers=True
)

# # model.output = SegmentationLabelsOutput(visualize=True)
# # model.serve()
trainer = Trainer(max_epochs=10, accelerator='cpu')
predictions = trainer.predict(model, datamodule=datamodule, output=FiftyOneSegmentationLabelsOutput())

predictions = list(chain.from_iterable(predictions))

# flatten batches
# Map filepaths to predictions
predictions = {p["filepath"]: p["predictions"] for p in predictions}

# Add predictions to FiftyOne dataset
dataset.set_values("flash_predictions", predictions, key_field="filepath")

# 8 Analyze predictions in the App
# session = fo.launch_app(dataset)
# session = fo.launch_app(dataset, remote=True, address='kp141-124', port=49162)
# session.wait()
# session.close()

# export the dataset
labels_path = "/projects/segment-sideview/results"
label_field = "flash_predictions"

dataset.export(
    dataset_type=fo.types.ImageSegmentationDirectory,
    labels_path=labels_path,
    label_field=label_field
)