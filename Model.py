import tensorflow as tf
import pandas as pd
import numpy as np
import torch
import os
import io
import zipfile
from google.cloud import storage
from google.colab import auth
import matplotlib.pyplot as plt
import time
from google.colab import drive
import gc


os.environ['GOOGLE_CLOUD_PROJECT'] = 'norse-analyst-457504-b2'
auth.authenticate_user()
try:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('feature_map')
    print(f"Bucket accessed successfully: {bucket.name}")
except Exception as e:
    raise Exception(f"Failed to access GCS bucket 'feature_map': {e}")


def mount_drive(max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            drive.mount('/content/drive', force_remount=True)
            print("Mounted Google Drive successfully")
            return True
        except Exception as e:
            print(f"Drive mount attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return False

if not mount_drive():
    raise Exception("Failed to mount Google Drive after multiple attempts")


dataset_zip_path = "/content/drive/MyDrive/dataset.zip"
dataset_extract_path = "/content/dataset"
max_retries = 3
delay = 5

def unzip_dataset():
    for attempt in range(max_retries):
        try:
            if not os.path.exists(dataset_extract_path):
                os.makedirs(dataset_extract_path)
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_extract_path)
            print(f"Extracted dataset to {dataset_extract_path}")
            return True
        except (OSError, zipfile.BadZipFile) as e:
            print(f"Unzip attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                print("Remounting Google Drive and retrying...")
                if not mount_drive():
                    raise Exception("Failed to remount Google Drive")
                time.sleep(delay)
    return False

if not unzip_dataset():
    raise Exception(f"Failed to unzip {dataset_zip_path} after {max_retries} attempts")

# Clean sample data directory
if os.path.exists('/content/sample_data'):
    os.system('rm -rf /content/sample_data')
    print("Removed /content/sample_data")

# Load metadata
bucket_name = 'feature_map'
metadata_path = '/content/dataset/dataset/metadata.csv'
try:
    metadata = pd.read_csv(metadata_path)
except Exception as e:
    raise Exception(f"Failed to load metadata from {metadata_path}: {e}")

# Define categories and validate
major_categories = [
    'Gan_inpainting', 'affine_transformation', 'background_change', 'blur', 'brightness_adjustment',
    'color_adjustment', 'contrast_adjustment', 'cropping', 'edge_enhancement', 'flipping',
    'hue_adjustment', 'inpainting', 'noise_addition', 'object_removal', 'object_insertion',
    'original', 'highlight_addition', 'perspective_transform', 'retouching', 'rotation',
    'saturation_adjustment', 'shadow_addition', 'sharpening', 'text_overlay', 'texture_overlay',
    'warping', 'splicing', 'original'
]
if not metadata['label'].isin(range(len(major_categories))).all():
    raise ValueError(f"Labels in metadata.csv contain values outside 0–{len(major_categories)-1}.")

# Process splits
metadata['split'] = metadata['image_path'].apply(lambda x: x.split('/')[0].lower())
valid_splits = ['train', 'val', 'test']
if not metadata['split'].isin(valid_splits).all():
    raise ValueError(f"Invalid splits found: {metadata['split'].unique()}.")

# Clean invalid mask paths, except for label=15 (original)
original_len = len(metadata)
# Allow NaN/empty mask_path for label=15
mask_invalid = (metadata['mask_path'].isna() | (metadata['mask_path'] == '')) & (metadata['label'] != 15)
dropped = metadata[mask_invalid]
if len(dropped) > 0:
    print(f"Dropped {len(dropped)} rows due to invalid mask_path (excluding label=15). Sample dropped rows:")
    print(dropped[['image_path', 'mask_path', 'label']].head())
metadata = metadata[~mask_invalid]

# Create dataset splits
train_df = metadata[metadata['split'] == 'train']
val_df = metadata[metadata['split'] == 'val']
test_df = metadata[metadata['split'] == 'test']

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("One or more splits (train/val/test) are empty.")

# Feature and mask loading functions
def load_feature_map(file_path):
    file_path_str = file_path.numpy().decode('utf-8')
    blob_path = file_path_str.replace(f'gs://{bucket_name}/', '')
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        print(f"Skipping missing file: {file_path_str}")
        return tf.zeros((256, 256, 512), dtype=tf.float32), tf.constant(False)
    try:
        pt_bytes = blob.download_as_bytes()
        feature_map = torch.load(io.BytesIO(pt_bytes))
        feature_map = tf.transpose(feature_map, perm=[1, 2, 0])
        feature_map = tf.cast(feature_map, tf.float32)
        feature_map = tf.ensure_shape(feature_map, [256, 256, 512])
        return feature_map, tf.constant(True)
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")
        return tf.zeros((256, 256, 512), dtype=tf.float32), tf.constant(False)

def load_mask(file_path, label):
    # For label=15 (original), return a zero mask
    if label == 15:
        return tf.zeros((256, 256, 1), dtype=tf.float32), tf.constant(True)
    file_path_str = file_path.numpy().decode('utf-8')
    if not tf.io.gfile.exists(file_path_str):
        print(f"Skipping missing mask file: {file_path_str}")
        return tf.zeros((256, 256, 1), dtype=tf.float32), tf.constant(False)
    try:
        file_content = tf.io.read_file(file_path_str)
        mask = tf.image.decode_png(file_content, channels=1)
        mask = tf.image.resize(mask, [256, 256])
        mask = tf.cast(mask > 0, tf.float32)
        mask = tf.ensure_shape(mask, [256, 256, 1])
        # Debug mask sparsity
        foreground_pixels = tf.reduce_sum(mask)
        if foreground_pixels < 100:  # Arbitrary threshold for sparsity
            print(f"Sparse mask detected: {file_path_str}, foreground pixels: {foreground_pixels}")
        return mask, tf.constant(True)
    except Exception as e:
        print(f"Error loading mask {file_path_str}: {e}")
        return tf.zeros((256, 256, 1), dtype=tf.float32), tf.constant(False)

# Dataset creation
def create_dataset(df, batch_size):
    skipped_samples = 0
    total_samples = len(df)
    split_name = df['split'].iloc[0]

    def verify_path(row):
        nonlocal skipped_samples
        category = major_categories[row['label']]
        prefix = 'Gan_' if category == 'Gan_inpainting' else category.split('_')[0].lower() + '_'
        file_id = os.path.basename(row['image_path']).split('_')[-1].split('.')[0]
        feature_path = f'{row["split"].capitalize()}/{prefix}{file_id}.pt'
        blob = storage_client.bucket(bucket_name).blob(feature_path)
        if not blob.exists():
            print(f"Missing feature map: gs://{bucket_name}/{feature_path}")
            skipped_samples += 1
            return None
        return f'gs://{bucket_name}/{feature_path}'

    feature_paths = df.apply(verify_path, axis=1)
    valid_rows = feature_paths.notna()
    df_valid = df[valid_rows].copy()
    feature_paths = feature_paths[valid_rows].values
    mask_paths = df_valid['mask_path'].values
    labels = df_valid['label'].astype(np.int32).values

    # Logging for debugging
    print(f"Dataset split: {split_name}")
    print(f"Total samples: {total_samples}, Skipped samples: {skipped_samples}, Valid samples: {len(df_valid)}")
    if len(df_valid) > 0:
        print(f"First 5 valid feature paths:")
        for path in feature_paths[:5]:
            print(path)
        print(f"First 5 mask paths:")
        for path in mask_paths[:5]:
            print(path)
        print(f"First 5 labels: {labels[:5]}")
    else:
        print("No valid samples found in this split!")

    feature_ds = tf.data.Dataset.from_tensor_slices(feature_paths).map(
        lambda x: tf.py_function(func=load_feature_map, inp=[x], Tout=[tf.float32, tf.bool]),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda feature, valid: (tf.ensure_shape(feature, [256, 256, 512]), valid),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    mask_ds = tf.data.Dataset.from_tensor_slices((mask_paths, labels)).map(
        lambda x, lbl: tf.py_function(func=load_mask, inp=[x, lbl], Tout=[tf.float32, tf.bool]),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda mask, valid: (tf.ensure_shape(mask, [256, 256, 1]), valid),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((feature_ds, mask_ds, label_ds)).filter(
        lambda feature_tuple, mask_tuple, label: feature_tuple[1] & mask_tuple[1]
    ).map(
        lambda feature_tuple, mask_tuple, label: (
            tf.ensure_shape(feature_tuple[0], [256, 256, 512]),
            (
                tf.ensure_shape(label, []),
                tf.ensure_shape(mask_tuple[0], [256, 256, 1])
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    gc.collect()
    return dataset


BATCH_SIZE = 4  

# Create datasets
train_ds = create_dataset(train_df, BATCH_SIZE)
val_ds = create_dataset(val_df, BATCH_SIZE)
test_ds = create_dataset(test_df, BATCH_SIZE)

# Loss functions and metrics
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    return 1 - (2 * intersection + 1e-6) / denominator

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    return (2 * intersection + 1e-6) / denominator

# Model definition
def transformer_bottleneck(x, num_heads=4, projection_dim=256):
    height = x.shape[1]
    width = x.shape[2]
    channels = x.shape[3]

    x_proj = tf.keras.layers.Conv2D(projection_dim, 1, padding='same')(x)
    x_flat = tf.keras.layers.Reshape((height * width, projection_dim))(x_proj)

    x_att = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads
    )(x_flat, x_flat)

    x = tf.keras.layers.Add()([x_att, x_flat])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x_ff = tf.keras.layers.Dense(projection_dim, activation='relu')(x)
    x = tf.keras.layers.Add()([x_ff, x])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Reshape((height, width, projection_dim))(x)
    return x

def unet_plus_plus(input_tensor, gan_refinement=False):
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x1)
    p1 = tf.keras.layers.MaxPooling2D(2)(x1)

    x2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    x2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x2)
    p2 = tf.keras.layers.MaxPooling2D(2)(x2)

    x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)

    bottleneck = transformer_bottleneck(x3)

    u2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(bottleneck)
    u2 = tf.keras.layers.Concatenate()([u2, x2])
    u2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(u2)

    u1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(u2)
    u1 = tf.keras.layers.Concatenate()([u1, x1])
    u1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(u1)

    mask = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(u1)
    if gan_refinement:
        residual = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(mask)
        residual = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(residual)
        mask = tf.keras.layers.Add()([mask, residual])

    return mask

def build_model():
    inputs = tf.keras.Input(shape=(256, 256, 512))
    cls = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(inputs)
    cls = tf.keras.layers.GlobalAveragePooling2D()(cls)
    cls_out = tf.keras.layers.Dense(len(major_categories), activation='softmax', name='cls_out')(cls)
    mask_out = unet_plus_plus(inputs, gan_refinement=True)
    mask_out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='mask_out')(mask_out)
    model = tf.keras.Model(inputs=inputs, outputs=[cls_out, mask_out])
    return model

# Compile model
model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / np.sqrt(2)),
    loss={
        'cls_out': 'sparse_categorical_crossentropy',
        'mask_out': lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    },
    metrics={
        'cls_out': 'accuracy',
        'mask_out': [iou, dice]
    }
)
model.summary()

# Callbacks
class CombinedLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['val_total_loss'] = logs.get('val_cls_out_loss', 0) + logs.get('val_mask_out_loss', 0)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'gs://{bucket_name}/checkpoints/weights_epoch_{{epoch:02d}}.weights.h5',
    save_weights_only=True,
    save_freq='epoch'
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_total_loss',
    patience=5,
    restore_best_weights=True
)

custom_combined_loss_cb = CombinedLossCallback()

# Training
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, custom_combined_loss_cb, earlystop_cb]
)

# Save final model
model.save_weights(f'gs://{bucket_name}/checkpoints/best_weights.weights.h5')

# Evaluation
test_metrics = model.evaluate(test_ds, return_dict=True)
print("Test Metrics:")
print(f"Classification Accuracy: {test_metrics['cls_out_accuracy']:.4f}")
print(f"Mask IoU: {test_metrics['mask_out_iou']:.4f}")
print(f"Mask Dice: {test_metrics['mask_out_dice']:.4f}")

# Visualization
def display_correct_predictions(model, test_df, num_samples=5):
    correct_samples = []
    manipulation_classes = {i: major_categories[i] for i in range(len(major_categories))}

    for _, row in test_df.iterrows():
        image_path = f'{dataset_extract_path}/dataset/{row["image_path"]}'
        mask_path = f'{dataset_extract_path}/dataset/{row["mask_path"]}'
        feature_path = f'gs://{bucket_name}/{row["split"].capitalize()}/' + \
                      ('Gan_' if major_categories[row['label']] == 'Gan_inpainting' else
                       major_categories[row['label']].split('_')[0].lower() + '_') + \
                      os.path.basename(row['image_path']).split('_')[-1].split('.')[0] + '.pt'
        true_label = row['label']

        try:
            feature_map, is_valid = load_feature_map(tf.constant(feature_path))
            if not is_valid:
                continue
            feature_map = tf.expand_dims(feature_map, 0)
            cls_pred, mask_pred = model.predict(feature_map, verbose=0)
            pred_label = np.argmax(cls_pred[0])

            if pred_label == true_label:
                image = tf.image.decode_jpeg(tf.io.read_file(image_path))
                image = tf.image.resize(image, [256, 256]) / 255.0
                mask, mask_valid = load_mask(tf.constant(mask_path), tf.constant(true_label, dtype=tf.int32))
                if not mask_valid:
                    continue
                correct_samples.append({
                    'image': image,
                    'true_mask': mask,
                    'pred_mask': mask_pred[0],
                    'true_label': true_label,
                    'pred_label': pred_label
                })

            if len(correct_samples) >= num_samples:
                break
        except Exception as e:
            print(f"Error processing sample {image_path}: {e}")
            continue

    if correct_samples:
        plt.figure(figsize=(15, 3 * len(correct_samples)))
        for i, sample in enumerate(correct_samples):
            plt.subplot(len(correct_samples), 4, i * 4 + 1)
            plt.imshow(sample['image'])
            plt.title(f'Original Image\nClass: {manipulation_classes[sample["true_label"]]}')
            plt.axis('off')

            plt.subplot(len(correct_samples), 4, i * 4 + 2)
            plt.imshow(sample['true_mask'].numpy().squeeze(), cmap='gray')
            plt.title('Ground-Truth Mask')
            plt.axis('off')

            plt.subplot(len(correct_samples), 4, i * 4 + 3)
            plt.imshow(sample['pred_mask'].squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.subplot(len(correct_samples), 4, i * 4 + 4)
            plt.text(0.5, 0.5, f'Predicted: {manipulation_classes[sample["pred_label"]]}', ha='center', va='center')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('correct_predictions.png')
        plt.close()
    else:
        print("No correctly classified samples found to display.")

print("Displaying sample outputs for correctly classified test samples...")
display_correct_predictions(model, test_df)
