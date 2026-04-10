"""
NuScenes Noise Dataset V2 - 支持丢帧、外参扰动、遮挡测试的 Dataset

基于 CustomNuScenesDatasetV2 扩展，复用 robust_benchmark 工具箱生成的 noise pkl 数据。

噪声类型和参数：
- 丢帧 (drop_frames): drop_ratio=[10,20,...,90], drop_type='discrete'/'consecutive'
- 外参扰动 (extrinsics_noise): noise_level='L1'~'L4', noise_scope='single'/'all'
- 遮挡 (mask_noise): 从 noise pkl 读取 mask_id，通过 Pipeline 叠加

使用方式：
    在 config 中设置 dataset_type='NuScenesNoiseDatasetV2' 并传入噪声参数。
"""
import copy
import numpy as np
from mmdet.datasets import DATASETS
from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2
import mmcv


@DATASETS.register_module()
class NuScenesNoiseDatasetV2(CustomNuScenesDatasetV2):
    """支持噪声注入的 NuScenes Dataset V2。

    Args:
        noise_nuscenes_ann_file (str): 噪声 pkl 文件路径
        drop_frames (bool): 是否启用丢帧
        drop_ratio (int): 丢帧比例, 10~90
        drop_type (str): 丢帧类型, 'discrete' 或 'consecutive'
        extrinsics_noise (bool): 是否启用外参扰动
        extrinsics_noise_level (str): 扰动等级, 'L1'~'L4'
        extrinsics_noise_scope (str): 扰动范围, 'single'(单摄像头) 或 'all'(多摄像头)
    """

    def __init__(self,
                 noise_nuscenes_ann_file='',
                 drop_frames=False,
                 drop_ratio=0,
                 drop_type='discrete',
                 extrinsics_noise=False,
                 extrinsics_noise_level='L3',
                 extrinsics_noise_scope='single',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.drop_frames = drop_frames
        self.drop_ratio = drop_ratio
        self.drop_type = drop_type
        self.extrinsics_noise = extrinsics_noise
        self.extrinsics_noise_level = extrinsics_noise_level
        self.extrinsics_noise_scope = extrinsics_noise_scope

        # 加载噪声数据
        if noise_nuscenes_ann_file and (self.drop_frames or self.extrinsics_noise):
            print(f'[NuScenesNoiseDatasetV2] Loading noise data from {noise_nuscenes_ann_file}')
            noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
            self.noise_camera_data = noise_data.get('camera', {})
            self.noise_lidar_data = noise_data.get('lidar', {})
        else:
            self.noise_camera_data = {}
            self.noise_lidar_data = {}

        # 打印噪声设置
        self._print_noise_settings()

    def _print_noise_settings(self):
        print('=' * 60)
        print('[NuScenesNoiseDatasetV2] Noise Settings:')
        if self.drop_frames:
            print(f'  Drop frames: ratio={self.drop_ratio}%, type={self.drop_type}')
        if self.extrinsics_noise:
            print(f'  Extrinsics noise: level={self.extrinsics_noise_level}, scope={self.extrinsics_noise_scope}')
        if not self.drop_frames and not self.extrinsics_noise:
            print('  No noise (baseline)')
        print('=' * 60)

    def prepare_input_dict(self, info):
        """重写父类方法，在数据准备时注入噪声。"""
        input_dict = super().prepare_input_dict(info)

        if not self.modality['use_camera']:
            return input_dict

        # ---- 丢帧处理 ----
        if self.drop_frames and self.noise_camera_data:
            image_paths = input_dict.get('img_filename', [])
            new_image_paths = []
            for img_path in image_paths:
                file_name = img_path.split('/')[-1]
                if file_name in self.noise_camera_data:
                    drop_info = self.noise_camera_data[file_name]['noise'].get('drop_frames', {})
                    ratio_info = drop_info.get(self.drop_ratio, {})
                    type_info = ratio_info.get(self.drop_type, {})
                    if type_info.get('stuck', False):
                        replace_file = type_info.get('replace', '')
                        if replace_file:
                            new_path = img_path.replace(file_name, replace_file)
                            new_image_paths.append(new_path)
                            continue
                new_image_paths.append(img_path)
            input_dict['img_filename'] = new_image_paths

        # ---- 外参扰动处理 ----
        if self.extrinsics_noise and self.noise_camera_data:
            noise_key_prefix = f'{self.extrinsics_noise_level}_{self.extrinsics_noise_scope}'
            rot_key = f'{noise_key_prefix}_noise_sensor2lidar_rotation'
            trans_key = f'{noise_key_prefix}_noise_sensor2lidar_translation'

            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = input_dict.get('cam2img', input_dict.get('cam_intrinsic', []))

            for i, (cam_type, cam_info) in enumerate(info['cams'].items()):
                file_name = cam_info['data_path'].split('/')[-1]

                if file_name in self.noise_camera_data:
                    extr_noise = self.noise_camera_data[file_name]['noise'].get('extrinsics_noise', {})
                    if rot_key in extr_noise and trans_key in extr_noise:
                        # 使用含噪声的外参
                        sensor2lidar_rotation = extr_noise[rot_key]
                        sensor2lidar_translation = extr_noise[trans_key]
                    else:
                        # 如果该级别噪声不存在，回退到原始外参
                        sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                        sensor2lidar_translation = cam_info['sensor2lidar_translation']
                else:
                    sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                    sensor2lidar_translation = cam_info['sensor2lidar_translation']

                # 重新计算 lidar2img 矩阵
                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict['lidar2img'] = lidar2img_rts
            input_dict['lidar2cam'] = lidar2cam_rts

        # ---- 遮挡信息传递（给 Pipeline 使用） ----
        # 将 noise_camera_data 通过 input_dict 传给 pipeline
        input_dict['noise_camera_data'] = self.noise_camera_data

        return input_dict

    def get_data_info(self, index):
        """重写 get_data_info，在 prepare_input_dict 已处理噪声的基础上进行。"""
        info = self.data_infos[index]
        input_dict = self.prepare_input_dict(info)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        if not self.test_mode and self.mono_cfg is not None:
            if input_dict is None:
                return None
            info = self.data_infos[index]
            img_ids = []
            for cam_type, cam_info in info['cams'].items():
                img_ids.append(cam_info['sample_data_token'])

            from mmcv.parallel import DataContainer as DC
            mono_input_dict = []; mono_ann_index = []
            for i, img_id in enumerate(img_ids):
                tmp_dict = self.mono_dataset.getitem_by_datumtoken(img_id)
                if tmp_dict is not None:
                    if self.filter_crowd_annotations(tmp_dict):
                        mono_input_dict.append(tmp_dict)
                        mono_ann_index.append(i)

            if len(mono_ann_index) == 0:
                return None

            mono_ann_index = DC(mono_ann_index, cpu_only=True)
            input_dict['mono_input_dict'] = mono_input_dict
            input_dict['mono_ann_idx'] = mono_ann_index
        return input_dict
