"""
遮挡 mask 叠加 Pipeline

复用 robust_benchmark Toolkit 中的遮挡模拟方案：
- 从 noise pkl 中读取每帧每个摄像头分配的 mask_id
- 加载对应 mask 图片，通过 alpha 混合叠加到原始图像
- 支持 mask_exp 参数控制遮挡强度等级 (S1=1.0, S2=2.0, S3=3.0, S4=5.0)
"""
import os
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMaskMultiViewImageFromFiles(object):
    """带遮挡 mask 的多视角图像加载 Pipeline。

    Args:
        to_float32 (bool): 是否转为 float32
        color_type (str): 图像读取颜色类型
        noise_nuscenes_ann_file (str): 噪声 pkl 文件路径
        mask_file (str): mask 图片目录路径  
        mask_exp (float): 遮挡强度指数
            S1=1.0 (轻微), S2=2.0 (中等), S3=3.0 (严重), S4=5.0 (极端)
    """

    def __init__(self, 
                 to_float32=False, 
                 color_type='unchanged',
                 noise_nuscenes_ann_file='', 
                 mask_file='',
                 mask_exp=3.0):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.mask_exp = mask_exp
        self.mask_file = mask_file

        # 加载噪声数据
        if noise_nuscenes_ann_file:
            noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
            self.noise_camera_data = noise_data.get('camera', {})
        else:
            self.noise_camera_data = {}

        # 预加载所有 mask 图片
        self.mask_cache = {}
        if mask_file and os.path.isdir(mask_file):
            for i in range(1, 17):
                mask_path = os.path.join(mask_file, f'mask_{i}.jpg')
                if os.path.exists(mask_path):
                    self.mask_cache[i] = mmcv.imread(mask_path, self.color_type)
            print(f'[LoadMaskMultiViewImageFromFiles] Loaded {len(self.mask_cache)} masks, exp={mask_exp}')

    def put_mask_on_img(self, img, mask):
        """将 mask 叠加到图像上。
        
        复用 robust_benchmark Toolkit 的遮挡模拟逻辑。
        alpha = (mask/255)^exp，exp 越大遮挡越严重。
        """
        h, w = img.shape[:2]
        mask = np.rot90(mask.copy())
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask.astype(np.float64) / 255.0
        alpha = np.power(alpha, self.mask_exp)
        img_with_mask = alpha * img.astype(np.float64) + (1 - alpha) * mask.astype(np.float64)
        return img_with_mask.astype(img.dtype)

    def __call__(self, results):
        """读取多视角图像并叠加遮挡 mask。"""
        filename = results['img_filename']

        img_lists = []
        for name in filename:
            single_img = mmcv.imread(name, self.color_type)

            # 查找该图像对应的 mask_id
            noise_index = name.split('/')[-1]
            if noise_index in self.noise_camera_data and self.mask_cache:
                mask_info = self.noise_camera_data[noise_index]['noise'].get('mask_noise', {})
                mask_id = mask_info.get('mask_id', None)
                if mask_id and mask_id in self.mask_cache:
                    single_img = self.put_mask_on_img(single_img, self.mask_cache[mask_id])

            img_lists.append(single_img)

        if self.to_float32:
            img_lists = [img.astype(np.float32) for img in img_lists]

        results['filename'] = filename
        results['img'] = img_lists
        results['img_shape'] = [img.shape for img in img_lists]
        results['ori_shape'] = [img.shape for img in img_lists]
        results['pad_shape'] = [img.shape for img in img_lists]
        num_channels = 1 if len(img_lists[0].shape) < 3 else img_lists[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(to_float32={self.to_float32}, "
                f"mask_exp={self.mask_exp})")
