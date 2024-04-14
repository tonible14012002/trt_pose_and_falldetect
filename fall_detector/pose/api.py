import numpy as np
import trt_pose
import cv2
import trt_pose.models
import json
import torch
import torch2trt
from torch2trt import TRTModule
import os

DEFAULT_MODEL_PATH = 'models/pose/'

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (3, 5): "m",
    (2, 2): "c",
    (4, 6): "c",
    (1, 2): "y",
    (1, 7): "m",
    (2, 8): "c",
    (7, 8): "y",
    (7, 9): "m",
    (9, 11): "m",
    (8, 10): "c",
    (10, 12): "c",
}


def is_valid_size(w, h):
    # Detection size should be divisible by 32.
    return w % 32 == 0 and h % 32 == 0


def cast_cv2_img_to_tf_tensor(image, size):
    """
    image: cv2 image
    size: (x, y)
    """
    return image

def load_model():
    return None


# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, EDGES, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        kx, ky, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 1, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        x1, y1, c1 = shaped[p1]
        x2, y2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4
            )

class Preprocessor:
    def __init__(self, sizeX, sizeY) -> None:
        self.sizeX = sizeX
        self.sizeY = sizeY

    def preprocess(self, image):
        return image
    
class PoseConfig:
    '''
    model = 'resnet' || 'densenet'
    '''
    def __init__(self, config_dir="./human_pose.json", model="resnet", sizeX=256, sizeY=256, model_file="", optimized_model_file=""):
        with open(config_dir, 'r') as f:
            human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.num_parts = len(human_pose['keypoints'])
        self.num_links = len(human_pose['skeleton'])
        self.model_type = model
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.file = model_file
        self.optimized_file = optimized_model_file

class ModelConvertor:
    storage_dir = None

    def __init__(self, config: PoseConfig) -> None:
        self._zero = torch.zeros((1, 3, config.sizeY, config.sizeX)).cuda()

        self.config = config
        self.model = trt_pose.models.resnet18_baseline_att(
            config.num_parts,
            2 * config.num_links,
        ).cuda().eval()
        self.optimized_model_path = '/'.join((self.storage_dir, self.config.optimized_file))
        self.base_model_path = '/'.join((self.storage_dir, self.config.file))

    def convert(self):
        if self.storage_dir == None:
            raise Exception('must call `set_save_dir` before execute this method')
        self.model.load_state_dict(torch.load(self.base_model_path))
        model_trt = torch2trt.torch2trt(self.model, [self._zero], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), self.optimized_model_path)
        return model_trt
    
    def set_save_dir(self, dir):
        self.storage_dir = dir
    
    def is_converted(self):
        return os.path.exists(self.optimized_model_path) == False
    
    def get_optimized_path(self):
        if self.is_converted():
            return self.optimized_model_path

class PoseModelLoader():
    def __init__(self):
        self.model_trt = TRTModule()

    def load(self, optimized_model_path):
        self.model_trt.load_state_dict(torch.load(optimized_model_path))
        return self.model_trt

class PoseEstimator:
    def __init__(self, config: PoseConfig, model: TRTModule) -> None:
        # assert is_valid_size(sizeX, sizeY), "Invalid size for detection model."
        pass

    def load_model(self):
        # self.model = tf_hub.load(
        #     "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1"
        # ).signatures["serving_default"]
        self.model = None

    def cast_to_tf_tensor(self, image):
        """
        cast cv2 image to tf tensor with resized to detection size
        """
        return cast_cv2_img_to_tf_tensor(image, self.size)

    def detect(self, pose_input, body_only=True):
        """
        pose_input = tensor image
        body_only: bool -> cut eyes, ears
        return: tensor (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        """
        # assert self.model is not None, "Model not loaded."
        # results = self.model(pose_input)
        # keypoints = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
        # if body_only:
        #     keypoints = tf.concat(
        #         [keypoints[:, :1, :], keypoints[:, 5:, :]], axis=1
        #     )
        # self.poses = keypoints
        # return keypoints
        return []

    def get_poses(self):
        """
        get current state poses
        """
        # return self.poses
        # poses = self.poses[:, :, [1, 0, 2]]  # x, y, confidence
        # print('result of get_poses, ', type(poses), 'with shape: ',  poses.shape)
        # return poses
        return []

    def filter_poses(self, threshold=0.2):
        pass
        # assert self.poses is not None
        # self.poses
        # scores = self.poses[:, :, 2]
        # mean_score_each = tf.reduce_mean(scores, axis=1)
        # mask_above_threshold = mean_score_each > threshold

        # self.poses = tf.boolean_mask(
        #     self.poses, mask_above_threshold, axis=0
        # ).numpy()
        # print('filtering poses with input: ', type(self.poses), 'with shape:', self.poses.shape)
    
def init_pose_esimation(model_type='densenet'):
    if model_type == 'densenet':
        config = PoseConfig(
            config_dir='./human_pose.json',
            model='densenet',
            sizeX=256,
            sizeY=256,
            model_file='densenet121_baseline_att_256x256_B_epoch_160.pth',
            optimized_model_file='densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        )
    else: # restnet
        config = PoseConfig(
            config_dir='./human_pose.json',
            model='resnet',
            sizeX=224,
            sizeY=224,
            model_file='resnet18_baseline_att_224x224_A_epoch_249.pth',
            optimized_model_file='resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        )

    convertor = ModelConvertor(config=config)

    convertor.set_save_dir('./models/pose')
    if not convertor.is_converted():
        convertor.convert()

    model_path = convertor.get_optimized_path()
    loader = PoseModelLoader()
    model_trt = loader.load(model_path)

    pose_estimator = PoseEstimator(config=config, model=model_trt)

    return pose_estimator
    
