import tensorflow as tf


def get_tokenizers(model_name="ted_hrlr_translate_pt_en_converter"):
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=".",
        cache_subdir="",
        extract=True,
    )
