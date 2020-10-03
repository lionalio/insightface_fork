from insightface.deploy import *
import argparse
import cv2
import sys
import os
import numpy as np

# Legacy
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)


#model = face_model.FaceModel(model='../../model-r100-ii/model,0')

def image_to_feature(path):
    '''
    Desc:
        Convert image to features

    Args:
        str: path to the image of candidate

    Ret:
        list: vector of features
    '''
    img_1 = cv2.imread(path)
    img_1 = model.get_input(img_1)
    f1 = model.get_feature(img_1)
    return f1


def enroll(img_path, path_matrix):
    '''
    Desc:
        Add the new candidate to the path of numpy object
        Convert image to features, then check if the subject
        has already existed. If data is empty or the real new one,
        add it to the existing path
        ### TODO: Also add new subject to the database with name
        and index

    Args:
        str: path to the image of candidate
        str: path of numpy data file locally

    Ret:
        bool: False if not added. True if added
    '''
    f = image_to_feature(img_path)
    matrix_features = load_matrix_features(path_matrix)
    if len(matrix_features) == 0:
        print('Blank data! Adding new subject')
        matrix_features = np.append(matrix_features, [f], axis=0)
        save_matrix_features(path_matrix, matrix_features)
        return True
    else:
        idx = verify(img_path, path_matrix)
        if idx >= 0:
            print('Subject has already existed!')
            return False
        else:
            print('New subject. Adding to data')
            matrix_features = np.append(matrix_features, [f], axis=0)
            save_matrix_features(path_matrix, matrix_features)
            return True


def verify(img_path, path_matrix, threshold=0.5):
    '''
    Desc:
        Verify the candidate using the path of numpy object
        Also set the threshold for the decision
        ### TODO: the verification must be carried out in multiple
        matrix path (should be done in multiprocess if needed)

    Args:
        str: path to the image of candidate
        str: path of numpy data file locally
        float: threshold for decision (Default: 0.5)

    Ret:
        To the matched subject in the database if exist. Else
        return -1
    '''
    f = image_to_feature(img_path)
    matrix_features = load_matrix_features(path_matrix)
    list_scores = np.dot(f, matrix_features.T)
    max_score = np.amax(list_scores)
    max_idx = np.where(list_scores == max_score)[0][0]
    if max_score > threshold:
        return max_idx
    else:
        return -1


def delete(idx, path_matrix):
    '''
    Desc:
        Delete the subject with corresponding index in the
        numpy object. Then save the numpy object
        ### TODO: Also delete the corresponding subject
        row in the database

    Args:
        int: index of subject in database & numpy matrix
        str: path of numpy data file locally

    Ret:
        None
    '''
    matrix_features = load_matrix_features(path_matrix)
    ret = np.delete(matrix_features, idx, 0)
    save_matrix_features(path_matrix, ret)
    print('Successfully deleted index ', idx)


def update(idx, img_path, path_matrix):
    '''
    Desc:
        update the feature of subject to the path of numpy object
        using the updated image path. Then save the updated
        feature matrix to its corresponding path

    Args:
        int: index of the subject in the database
        str: path to the updated image
        str: path of numpy data file locall

    Ret:
        None
    '''
    f = image_to_feature(img_path)
    matrix_features = load_matrix_features(path_matrix)
    matrix_features[idx] = f
    save_matrix_features(path_matrix, matrix_features)
    print('Successfully update index {} into path: {}'.format(idx, path_matrix))


def save_matrix_features(path, matrix_features):
    '''
    Desc:
        Save the matrix of features to the path of numpy object
        ### TODO: check the saved subject in the DB

    Args:
        str: path of numpy data file locally
        array: numpy matrix of features

    Ret:
        None
    '''
    np.save(path, matrix_features)
    print('Successfully saved matrix into path: ', path)


def load_matrix_features(path):
    '''
    Desc:
        Load the matrix of features from path of numpy object

    Args:
        str: path of numpy data file locally

    Ret:
        2D array: loaded numpy matrix
    '''
    with open(path, 'rb') as f:
        ret = np.load(f)
    return ret


def create_data(data_name):
    '''
    Desc:
        Create a numpy data object with name data_name
        This function create the data only if it doesn't exist yet

    Args:
        str: path to save the data file locally

    Ret:
        None
    '''
    if os.path.exists(data_name):
        print('Data existed already')
    else:
        path_init=data_name #'data1.pkl'
        mat = np.empty(shape=[0, 512], dtype=float, order='C')
        save_matrix_features(path_init, mat)


name1='data1.npy'
create_data(name1)
print(len(load_matrix_features(name1)))
#enroll('data/Ben_Affleck.jpg', name1)
#enroll('data/Elton_John.jpg', name1)
#enroll('data/Madonna.jpg', name1)
#update(1, 'data/Elton_John2.jpg', name1)
delete(2, name1)
print(len(load_matrix_features(name1)))
