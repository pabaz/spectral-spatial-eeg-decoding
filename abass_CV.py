import os
import warnings
import tensorflow as tf

# === TensorFlow noise suppression ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.debugging.set_log_device_placement(False)
warnings.filterwarnings('ignore')

# === Usual imports ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import DepthwiseConv2D


def se_block(x, ratio=8):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])

class TimeEncoding(tf.keras.layers.Layer):
    def __init__(self, time_dim):
        super().__init__()
        self.positional = self.add_weight("pos_emb", shape=(1, 1, time_dim, 1), initializer="uniform", trainable=True)

    def call(self, x):
        return x + self.positional

def channel_attention_block(x, ratio=8):
    channel = x.shape[-1]
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(channel // ratio, activation='relu')(squeeze)
    excitation = Dense(channel, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channel))(excitation)
    return Multiply()([x, excitation])

def residual_se_block(x, filters, kernel_size=(3, 10)):
    shortcut = x
    x = SeparableConv2D(filters, kernel_size, padding='same', use_bias=False, activation='elu')(x)
    x = BatchNormalization()(x)
    x = se_block(x)
    x = channel_attention_block(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
    return Add()([shortcut, x])

def DeepConvNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = TimeEncoding(input_shape[2])(inputs)

    x = Conv2D(25, (1, 10), padding='same', activation='elu', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((1, 10), padding='same', depth_multiplier=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('elu')(x)

    x = Conv2D(25, (input_shape[0], 1), use_bias=False, activation='elu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(1, 3), strides=(1, 3))(x)
    x = Dropout(0.5)(x)

    for _ in range(3):
        x = residual_se_block(x, filters=50)
        x = AveragePooling2D(pool_size=(1, 3), strides=(1, 3))(x)
        x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=10, min_lr=1e-5, verbose=1)

subs = ['sub11', 'sub13', 'sub8', 'sub9']
base_path = r"C:\\shared\\abass"
stimulus_labels = ['Left', 'Right', 'Up', 'Down']
# === [above imports remain unchanged, skipping for brevity] ===

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=64, alpha=0.3, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X1 = self.X[indices]
        y1 = self.y[indices]
        indices2 = np.random.permutation(indices)
        X2 = self.X[indices2]
        y2 = self.y[indices2]
        l = np.random.beta(self.alpha, self.alpha, size=(X1.shape[0], 1, 1, 1)).astype(np.float32)
        X = l * X1 + (1 - l) * X2
        y = l[:, 0, 0, 0][:, None] * y1 + (1 - l[:, 0, 0, 0][:, None]) * y2
        return X, y

# === Main loop per subject ===
for sub in subs:
    print(f"\n==== Subject: {sub} ====")
    sub_dir = os.path.join(base_path, sub)
    data_path = os.path.join(sub_dir, f"{sub}_scalp_epochs_by_class.npz")
    data = np.load(data_path)
    timepoints = data['left'].shape[2]
    time_axis = np.linspace(-0.5, 0.5, timepoints)
    start_idx = np.argmin(np.abs(time_axis - (-0.5)))
    end_idx = np.argmin(np.abs(time_axis - 0.5))

    X_all, y_all = [], []
    min_trials = min([data[key].shape[0] for key in ['left', 'right', 'up', 'down']])

    for label_idx, key in enumerate(['left', 'right', 'up', 'down']):
        trials = np.transpose(data[key][:, :, start_idx:end_idx], (1, 0, 2))
        trials = trials[:min_trials]  # trim to match smallest class
        for i in range(trials.shape[0]):
            trials[i] = (trials[i] - np.mean(trials[i], axis=1, keepdims=True)) / np.std(trials[i], axis=1, keepdims=True)
        X_all.append(trials)
        y_all.extend([label_idx] * trials.shape[0])

    X = np.concatenate(X_all)
    y = np.array(y_all)
    X = X[..., np.newaxis]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_idx, test_idx in skf.split(X, y):
        print(f"\nTraining CV Fold {fold} for {sub}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_cat = to_categorical(y[train_idx])
        y_test_cat = to_categorical(y[test_idx])

        model = DeepConvNet(input_shape=X.shape[1:], num_classes=4)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        train_generator = MixupGenerator(X_train, y_train_cat, batch_size=64, alpha=0.3)
        model.fit(train_generator, epochs=150, validation_data=(X_test, y_test_cat),
                  callbacks=[reduce_lr], verbose=1)

        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = y[test_idx]

        report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=stimulus_labels, output_dict=True)).transpose()
        report_df.to_csv(os.path.join(sub_dir, f"{sub}_cv_fold{fold}_classification_report.csv"), float_format='%.2f')
        report_df.to_json(os.path.join(sub_dir, f"{sub}_cv_fold{fold}_classification_report.json"), orient='split', indent=2)

        cm = confusion_matrix(y_true, y_pred)
        np.save(os.path.join(sub_dir, f"{sub}_cv_fold{fold}_confusion_matrix.npy"), cm)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='jet', xticklabels=stimulus_labels, yticklabels=stimulus_labels)
        ax.set_title(f"DeepConvNet CV Fold {fold} - {sub}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, f"{sub}_cv_fold{fold}_confusion_matrix.png"))
        plt.close()

        fold += 1

    print(f"Results saved for subject {sub} in {sub_dir}")
