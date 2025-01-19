import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds
import sys

sys.path.append("libs/keras")  # hacky but ok for the sake of this exercise

import libs.keras.tokenizer as tokenizer_helper
from libs.keras.transformer import Transformer
from libs.keras.training import CustomSchedule, masked_loss, masked_accuracy
from libs.keras.translator import Translator, ExportTranslator, translate


def download_dataset(name="ted_hrlr_translate/pt_to_en"):
    examples, metadata = tfds.load(name, with_info=True, as_supervised=True)

    return examples["train"], examples["validation"]


tokenizer_model_name = "ted_hrlr_translate_pt_en_converter"
tokenizer_helper.get_tokenizers(tokenizer_model_name)
tokenizers = tf.saved_model.load(
    "ted_hrlr_translate_pt_en_converter_extracted/" + tokenizer_model_name
)


def prepare_batch(pt, en, max_tokens=128):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :max_tokens]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, : (max_tokens + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(ds: tf.data.Dataset, buffer_size=20000, batch_size=64, seed=0):

    return (
        ds
        # Randomizes the order of the dataset.
        # The buffer_size defines how many elements are loaded into memory for shuffling at a time.
        # A larger buffer results in better randomization but requires more memory.
        .shuffle(buffer_size, seed=seed)
        # Groups batch_size elements into a single batch
        .batch(batch_size)
        # Applies the prepare_batch function to each batch of data
        # tf.data.AUTOTUNE automatically determines the optimal number of threads to use
        # for the mapping operation to maximize performance
        .map(prepare_batch, tf.data.AUTOTUNE)
        # Preloads a number of batches while the model processes the current one
        # tf.data.AUTOTUNE dynamically adjusts the prefetch buffer size for the best throughput
        # based on system performance
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


if __name__ == "__main__":

    # Download data
    train_examples, val_examples = download_dataset()

    # Print some examples
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print("> Examples in Portuguese:")
        for pt in pt_examples.numpy():
            print(pt.decode("utf-8"))
        print()

        print("> Examples in English:")
        for en in en_examples.numpy():
            print(en.decode("utf-8"))

    # Get data batches
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    for (pt, en), en_labels in train_batches.take(1):
        break

    # Hyperparameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=dropout_rate,
    )

    # This line causes an error when executed with input
    # output = transformer((pt, en)); transformer.summary()

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    transformer.summary()

    transformer.fit(train_batches, epochs=4, validation_data=val_batches)

    # Test translator
    translator = Translator(tokenizers, transformer)

    sentence = "este é um problema que temos que resolver."
    ground_truth = "this is a problem we have to solve ."
    translate(translator, sentence, ground_truth)

    translator = ExportTranslator(translator)
    tf.saved_model.save(translator, export_dir="pt_en_translator")

    reloaded_translator = tf.saved_model.load("pt_en_translator")
    print(reloaded_translator(sentence).numpy())
