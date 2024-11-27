import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np
from keras.models import Model
from fawkes.align_face import aligner
from fawkes.utils import init_gpu, load_extractor, load_victim_model, preprocess, Faces, load_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def filter_image_paths(image_paths):
    new_image_paths = []
    new_images = []
    print(f"Processing {len(image_paths)} images...")
    
    for p in image_paths:
        try:
            img = load_image(p)
            if img is None:
                print(f"Warning: Could not load image {p}")
                continue
            new_image_paths.append(p)
            new_images.append(img)
        except Exception as e:
            print(f"Error processing image {p}: {str(e)}")
    
    print(f"Successfully processed {len(new_images)} images")
    return new_image_paths, new_images

def get_features(model, paths, ali, batch_size=16):
    if not paths:
        raise ValueError("No image paths provided for feature extraction")
    
    print(f"Extracting features from {len(paths)} images...")
    paths, images = filter_image_paths(paths)
    
    if not images:
        raise ValueError("No valid images found for feature extraction")
    
    faces = Faces(paths, images, ali, verbose=0, eval_local=True, no_align=True)
    faces = faces.cropped_faces
    
    if faces is None or len(faces) == 0:
        raise ValueError("No faces detected in the provided images")
    
    features = model.predict(faces, verbose=0)
    print(f"Successfully extracted features: shape {features.shape}")
    return features

def get_feature_extractor(base_model="low_extract", custom_weights=None):
    print(f"Loading feature extractor: {base_model}")
    try:
        base_model = load_extractor(base_model)
        features = base_model.layers[-1].output
        model = Model(inputs=base_model.input, outputs=features)

        if custom_weights is not None:
            print(f"Loading custom weights from: {custom_weights}")
            model.load_weights(custom_weights, by_name=True, skip_mismatch=True)
            
        return model
    except Exception as e:
        raise Exception(f"Error loading feature extractor: {str(e)}")

def get_class(data_dir):
    print(f"Processing directory: {data_dir}")
    folders_arr = data_dir.split(os.path.sep)
    print(f"Split path: {folders_arr}")
    for i in range(len(folders_arr)-1):
        if folders_arr[i+1] == 'face':
            class_name = folders_arr[i]
            return class_name
    return None


def get_facescrub_features(model, ali, dataset_path):
    print(f"\nProcessing FaceScrub dataset from: {dataset_path}")
    data_dirs = sorted(glob.glob(os.path.join(dataset_path, "*")))
    
    if not data_dirs:
        raise ValueError(f"No directories found in {dataset_path}")
    
    print(f"Found {len(data_dirs)} person directories")
    
    classes_train = []
    features_train = []
    classes_test = []
    features_test = []

    for data_dir in data_dirs:
        face_dir = os.path.join(data_dir, "face")
        if not os.path.exists(face_dir):
            print(f"Warning: face directory not found in {data_dir}")
            continue
            
        cls = get_class(face_dir)
        print(f"\nProcessing class: {cls}")
        
        all_pathes = sorted(glob.glob(os.path.join(face_dir, "*.jpg")))
        if not all_pathes:
            print(f"Warning: No jpg images found in {face_dir}")
            continue
            
        print(f"Found {len(all_pathes)} images")
        
        try:
            f = get_features(model, all_pathes, ali)
            
            test_len = int(0.3 * len(all_pathes))
            test_idx = random.sample(range(len(all_pathes)), test_len)

            f_test = f[test_idx]
            f_train = np.delete(f, test_idx, axis=0)
            
            features_train.append(f_train)
            classes_train.extend([cls] * len(f_train))
            features_test.append(f_test)
            classes_test.extend([cls] * len(f_test))
            
            print(f"Successfully processed {len(f_train)} training and {len(f_test)} test samples")
            
        except Exception as e:
            print(f"Error processing directory {face_dir}: {str(e)}")
            continue

    if not features_train or not features_test:
        raise ValueError("No features were successfully extracted")

    classes_train = np.asarray(classes_train)
    features_train = np.concatenate(features_train, axis=0)
    classes_test = np.asarray(classes_test)
    features_test = np.concatenate(features_test, axis=0)

    print(f"\nTotal samples extracted:")
    print(f"Training: {features_train.shape[0]} samples")
    print(f"Testing: {features_test.shape[0]} samples")

    return features_train, features_test, classes_train, classes_test

def main():
    try:
        print("Initializing GPU...")
        sess = init_gpu("0")
        print("Creating face aligner...")
        ali = aligner(sess)
        print("Loading feature extractor...")
        model = get_feature_extractor("low_extract", custom_weights=args.robust_weights)

        random.seed(10)
        print("Extracting features...", flush=True)
        X_train_all, X_test_all, Y_train_all, Y_test_all = get_facescrub_features(model, ali, args.facescrub_dir)

        val_people = args.names_list
        print(f"Processing people: {val_people}")

        base_dir = args.attack_dir

        for name in val_people:
            directory = os.path.join(base_dir, name, "face")
            print(f"\nProcessing directory: {directory}")
            
            if not os.path.exists(directory):
                print(f"Warning: Directory does not exist: {directory}")
                continue
                
            image_paths = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
            
            all_pathes_uncloaked = sorted([path for path in image_paths if args.unprotected_file_match in os.path.basename(path)])
            all_pathes_cloaked = sorted([path for path in image_paths if args.protected_file_match in os.path.basename(path)])

            print(f"Found {len(all_pathes_cloaked)} cloaked and {len(all_pathes_uncloaked)} uncloaked images")
            
            if len(all_pathes_cloaked) != len(all_pathes_uncloaked):
                print("Warning: Unequal number of cloaked and uncloaked images")
                continue

            f_cloaked = get_features(model, all_pathes_cloaked, ali)
            f_uncloaked = get_features(model, all_pathes_uncloaked, ali)

            random.seed(10)
            test_frac = 0.3
            test_idx = random.sample(range(len(all_pathes_cloaked)), int(test_frac * len(all_pathes_cloaked)))

            f_train_cloaked = np.delete(f_cloaked, test_idx, axis=0)
            f_test_cloaked = f_cloaked[test_idx]
            f_train_uncloaked = np.delete(f_uncloaked, test_idx, axis=0)
            f_test_uncloaked = f_uncloaked[test_idx]

            if args.classifier == "linear":
                print("Using Logistic Regression classifier")
                clf1 = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False)
                clf1 = make_pipeline(StandardScaler(), clf1)
            else:
                print("Using K-Nearest Neighbors classifier")
                clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

            idx_train = np.asarray([y != name for y in Y_train_all])
            idx_test = np.asarray([y != name for y in Y_test_all])
            print(f"Training samples: {np.sum(idx_train)}, Test samples: {np.sum(idx_test)}")

            # with cloaking
            X_train = np.concatenate((X_train_all[idx_train], f_train_cloaked))
            Y_train = np.concatenate((Y_train_all[idx_train], [name] * len(f_train_cloaked)))
            print(type(Y_train), Y_train)

            print("Training classifier...")
            clf1 = clf1.fit(X_train, Y_train)

            print("\nResults:")
            print("idx_test:", idx_test)
            print("Number of test samples:", len(X_test_all), len(Y_test_all))
            print("Test acc: {:.2f}".format(clf1.score(X_test_all[idx_test], Y_test_all[idx_test])))
            print("Train acc (user cloaked): {:.2f}".format(clf1.score(f_train_cloaked, [name] * len(f_train_cloaked))))
            print("Test acc (user cloaked): {:.2f}".format(clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
            print("Protection rate: {:.2f}".format(1-clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
            print(flush=True)

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print("\nDebug information:")
        print(f"Working directory: {os.getcwd()}")
        print(f"FaceScrub directory exists: {os.path.exists(args.facescrub_dir)}")
        print(f"Attack directory exists: {os.path.exists(args.attack_dir)}")
        raise

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', type=str,
                        help='the feature extractor', default='low_extract')
    parser.add_argument('--classifier', type=str,
                        help='the classifier', default='NN')
    parser.add_argument('--robust-weights', type=str, 
                        help='robust weights', default=None)
    parser.add_argument('--names-list', nargs='+', default=[], help="names of attacking users")
    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory', default="facescrub/download/")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory', default="facescrub_attacked/download/")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='high_cloaked.png')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()