import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from PIL import Image
import joblib
import scipy.stats
import scipy
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam

def describe_freq(freqs):
    mean = np.mean(freqs)
    std = np.std(freqs)
    maxv = np.amax(freqs)
    minv = np.amin(freqs)
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    mode = scipy.stats.mode(freqs, axis=None)[0]
    iqr = scipy.stats.iqr(freqs)
    
    return np.array([mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr])
    
def extract_features_audio(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    fft_results = np.abs(np.fft.rfft(audio)) 
    freqs = np.fft.rfftfreq(len(audio))
    statistical_features = describe_freq(fft_results)
    energy = np.sum(np.square(audio)) 
    rmse_value = np.sqrt(np.mean(np.square(audio)))
    zero_crossings = sum(librosa.zero_crossings(audio, pad=False))
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
    poly_features = librosa.feature.poly_features(S=librosa.stft(audio), order=1)
    oenv = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]

    features = np.hstack([statistical_features, [energy], rmse_value, [zero_crossings], [tempo], mfccs.mean(axis=1), poly_features.mean(axis=1), tempogram.mean(axis=1), [spec_centroid.mean()], [spectral_bandwidth.mean()], [spectral_contrast.mean()], [spectral_flatness.mean()], [spectral_rolloff.mean()]])
    features = np.real(features)
    return features

def extract_features_image(file_path):
    img = Image.open(file_path).convert('L').resize((128, 128))
    return np.array(img).flatten() 

def load_and_combine_features(base_path):
    audio_features, image_features, labels = [], [], []
    for label_folder in ['HC', 'PD']:
        folder_path = os.path.join(base_path, label_folder)
        file_ids = [f.split('.')[0] for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        for file_id in file_ids:
            print(file_id)
            try:
                audio_file_path = os.path.join(folder_path, f'{file_id}.wav')
                image_file_path = os.path.join(folder_path, f'{file_id}.png')
                audio_feat = extract_features_audio(audio_file_path)
                image_feat = extract_features_image(image_file_path)
                
                label = 0 if label_folder == 'HC' else 1  # 0 for Healthy Control, 1 for Parkinson's Disease
                audio_features.append(audio_feat)
                image_features.append(image_feat)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_id} in {label_folder}: {e}")
                
    return np.array(audio_features), np.array(image_features), np.array(labels)


def create_multi_input_model(input_shape_audio, input_shape_image):
    # Audio input branch
    audio_input = Input(shape=(input_shape_audio,), name='audio_input')
    audio_branch = Dense(96, activation='relu')(audio_input)
    audio_branch = Dropout(0.5)(audio_branch)
    audio_branch = Dense(48, activation='relu')(audio_branch)
    audio_branch = Dropout(0.2)(audio_branch)
    audio_branch = Dense(8, activation='relu')(audio_branch)

    # Image input branch
    image_input = Input(shape=(input_shape_image, input_shape_image, 1), name='image_input')
    image_branch = Conv2D(16, (3, 3), activation='relu')(image_input)
    image_branch = MaxPooling2D((2, 2))(image_branch)
    image_branch = Conv2D(32, (3, 3), activation='relu')(image_branch)
    image_branch = MaxPooling2D((2, 2))(image_branch)
    image_branch = Conv2D(48, (3, 3), activation='relu')(image_branch)
    image_branch = MaxPooling2D((2, 2))(image_branch)
    image_branch = Flatten()(image_branch)
    image_branch = Dense(16, activation='relu')(image_branch)
    image_branch = Dropout(0.5)(image_branch)  # Adjust dropout rate as needed

    # Combine branches
    combined = concatenate([audio_branch, image_branch], axis=-1)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)  # Adjust dropout rate as needed
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)  # Adjust dropout rate as needed
    output = Dense(2, activation='softmax')(combined)

    model = Model(inputs=[audio_input, image_input], outputs=output)
    return model
    
def save_features(audio_features, image_features, labels, filename):
    with open(filename, 'wb') as f:
        # Save a tuple containing the audio features, image features, and labels
        joblib.dump((audio_features, image_features, labels), f)

def load_features(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            # Load the file's contents
            audio_features, image_features, labels = joblib.load(f)
            return audio_features, image_features, labels
    else:
        # Return None for each expected output if the file doesn't exist
        return None, None, None

def main():

    train_path = 'C:/Users/jpgallego/Desktop/ParkinsonsDisease/CombinedTrain'
    test_path = 'C:/Users/jpgallego/Desktop/ParkinsonsDiseaseCombinedTest'
    train_features_file = 'C:/Users/jpgallego/Desktop/ParkinsonsDisease/train_features.pkl'
    test_features_file = 'C:/Users/jpgallego/Desktop/ParkinsonsDisease/test_features.pkl'
    image_dimension = 128

    loaded_train_data = load_features(train_features_file)
    if loaded_train_data[0] is not None and loaded_train_data[1] is not None and loaded_train_data[2] is not None:
        print("Train data loaded")
        X_train_audio, X_train_image, y_train = loaded_train_data
    else:
        X_train_audio, X_train_image, y_train = load_and_combine_features(train_path)
        save_features(X_train_audio, X_train_image, y_train, train_features_file)


    loaded_test_data = load_features(test_features_file)
    if loaded_test_data[0] is not None and loaded_test_data[1] is not None and loaded_test_data[2] is not None:
        print("Test data loaded")
        X_test_audio, X_test_image, y_test = loaded_test_data
    else:
        X_test_audio, X_test_image, y_test = load_and_combine_features(test_path)
        save_features(X_test_audio, X_test_image, y_test, test_features_file)

    # Scale and transform features independently for audio and image:
    scaler_audio = StandardScaler()
    scaler_image = StandardScaler()
    X_train_audio = scaler_audio.fit_transform(X_train_audio)
    X_train_image = scaler_image.fit_transform(X_train_image.reshape(X_train_image.shape[0], -1))  # Flatten image data for scaling
    X_test_audio = scaler_audio.transform(X_test_audio)
    X_test_image = scaler_image.transform(X_test_image.reshape(X_test_image.shape[0], -1))  # Flatten image data for scaling

    # Reshape image data back after scaling:
    X_train_image = X_train_image.reshape(-1, image_dimension, image_dimension, 1)
    X_test_image = X_test_image.reshape(-1, image_dimension, image_dimension, 1)

     # Dimensionality Reduction with PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_audio = pca.fit_transform(X_train_audio)
    X_test_audio = pca.transform(X_test_audio)
    
    print(len(X_train_audio[1]))
    print(image_dimension*image_dimension)

    # Convert labels to categorical:
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    multi_input_model = create_multi_input_model(len(X_train_audio[1]), image_dimension)
    multi_input_model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    
    # When calling fit on the model, provide the lists of audio and image features separately:
    history = multi_input_model.fit([X_train_audio, X_train_image], y_train, epochs=50, batch_size=64, validation_data=([X_test_audio, X_test_image], y_test), callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: print(f'Epoch {epoch + 1}: Training Loss {logs["loss"]}, Training Accuracy {logs["accuracy"]}'))])

    # Evaluation remains the same, feeding separate audio and image features into the model
    test_loss, test_accuracy = multi_input_model.evaluate([X_test_audio, X_test_image], y_test)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
    print(multi_input_model.summary())

if __name__ == '__main__':
    main()