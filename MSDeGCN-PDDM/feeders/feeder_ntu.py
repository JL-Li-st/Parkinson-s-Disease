#
# import os
# import glob
# from . import tools
# import pandas as pd
# import pickle
# import numpy as np
# import re
# from torch.utils.data import Dataset
#
#
# class Feeder(Dataset):
#     def __init__(self, data_folder, label_file_path, p_interval=1, split='train', random_choose=False, random_shift=False,
#                  random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
#                  bone=False, vel=False):
#         self.debug = debug
#         self.data_folder = data_folder
#         self.label_file_path = label_file_path
#         self.split = split
#         self.random_choose = random_choose
#         self.random_shift = random_shift
#         self.random_move = random_move
#         self.window_size = window_size if window_size > 0 else None
#         self.normalization = normalization
#         self.use_mmap = use_mmap
#         self.p_interval = p_interval
#         self.random_rot = random_rot
#         self.bone = bone
#         self.vel = vel
#
#         # Load the label file
#         self.label_df = pd.read_excel(label_file_path)
#
#         # Load all data files from the folder
#         self.data_files = glob.glob(os.path.join(data_folder, '*.csv'))
#         if len(self.data_files) == 0:
#             raise ValueError(f"No .csv files found in the folder: {data_folder}")
#
#         # Determine the minimum number of frames across all CSV files
#         self.min_frames = self.get_min_frames()
#
#         self.data_list = []
#         self.label_list = []
#         self.load_data()
#
#         if normalization:
#             self.get_mean_map()
#
#     def get_min_frames(self):
#         min_frames = float('inf')
#         for data_path in self.data_files:
#             csv_data = pd.read_csv(data_path, header=None)
#             num_frames = csv_data.shape[0]
#             if num_frames < min_frames:
#                 min_frames = num_frames
#         return min_frames
#
#     def load_data(self):
#         for data_path in self.data_files:
#             # Extract Participant ID and Event ID from the filename
#             filename = os.path.basename(data_path)
#             match = re.match(r'Pt(\d+)_(C|PD)_n_(\d+)\.csv.*', filename)
#             if not match:
#                 print(f"Skipping file with incorrect format: {filename}")
#                 continue
#
#             participant_id = int(match.group(1))
#             event_id = int(match.group(3))
#
#             # Find the corresponding row in the label dataframe
#             participant_filter = self.label_df['Participant ID number'] == participant_id
#             event_filter = self.label_df['Turn ID'] == event_id
#             row = self.label_df[participant_filter & event_filter]
#
#             if row.empty:
#                 print(f"No matching label found for participant {participant_id} and event {event_id}, skipping.")
#                 continue
#
#             pd_or_c = row['PD_or_C'].values[0]
#             if pd_or_c == 'C':
#                 label = np.array([0])  # Healthy individual
#             elif pd_or_c == 'PD':
#                 label = np.array([1])  # Patient with disease
#             else:
#                 print(f"Unexpected value in PD_or_C column: {pd_or_c}, skipping.")
#                 continue
#
#             # Load the data from the CSV file
#             csv_data = pd.read_csv(data_path, header=None)
#
#             # 数据集中没有无关列，直接限制帧数
#             csv_data = csv_data.iloc[:self.min_frames, :].values
#
#             # 检查列数是否正确（75列 = 25个关键点 × 3维）
#             if csv_data.shape[1] != 75:
#                 print(f"Unexpected number of columns in {filename} (expected 75, got {csv_data.shape[1]}), skipping.")
#                 continue
#
#             # 计算帧数和关键点数量
#             T = csv_data.shape[0]  # 时间帧数
#             V = 25  # 关键点数量（固定为 25）
#
#             # 重塑数据为 (T, V, 3)
#             data = csv_data.reshape((T, V, 3))
#
#             # 将数据和标签添加到列表
#             self.data_list.append(data)
#             self.label_list.append(label)
#
#         # 检查是否有有效数据
#         if len(self.data_list) == 0:
#             raise ValueError("No valid data files were loaded. Please check your dataset.")
#
#         # 转换列表为 numpy 数组
#         self.data = np.array(self.data_list)
#         self.label = np.array(self.label_list)
#
#     def get_mean_map(self):
#         data = self.data
#         N, T, V, C = data.shape
#         self.mean_map = data.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).mean(axis=0)
#         self.std_map = data.reshape((N * T * V, C)).std(axis=0).reshape((1, 1, V, C))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __iter__(self):
#         return self
#
#     def __getitem__(self, index):
#         data_numpy = self.data[index]  # 原始形状: (T, V, C)
#
#         # 扩展为 4D: 增加一个额外的维度以匹配 (C, T, V, M)
#         # 假设 M 表示帧中的人数，因为只有一个实例，所以 M = 1
#         data_numpy = np.expand_dims(data_numpy, axis=-1)  # 形状变为 (T, V, C, 1)
#         data_numpy = np.transpose(data_numpy, (2, 0, 1, 3))  # 重排顺序为 (C, T, V, M)
#
#         label = self.label[index]
#
#         # 计算有效帧的数量
#         valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
#
#         # 设置目标帧数为 509 帧
#         target_frame_num = 100
#
#         # 使用 auto_pading 函数将数据填充为 509 帧
#         data_numpy = tools.auto_pading(data_numpy, target_frame_num, random_pad=False)
#
#         # 调用 tools.valid_crop_resize 时，数据应是 4 维的
#         p_interval = self.p_interval if isinstance(self.p_interval, (list, tuple)) else [self.p_interval]
#         data_numpy = tools.valid_crop_resize(data_numpy, target_frame_num, p_interval, target_frame_num)
#
#         return data_numpy, label, index
#
#
#
#     def top_k(self, score, top_k):
#         rank = score.argsort()
#         hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
#         return sum(hit_top_k) * 1.0 / len(hit_top_k)
#
#
# def import_class(name):
#     components = name.split('.')
#     mod = __import__(components[0])
#     for comp in components[1:]:
#         mod = getattr(mod, comp)
#     return mod
#
import os
import glob
from . import tools
import pandas as pd
import pickle
import numpy as np
import re
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_folder, label_file_path, p_interval=1, split='train', random_choose=False,
                 random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        self.debug = debug
        self.data_folder = data_folder
        self.label_file_path = label_file_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size if window_size > 0 else None
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel

        # Load the label file
        self.label_df = pd.read_excel(label_file_path)

        # Load all data files from the folder
        self.data_files = glob.glob(os.path.join(data_folder, '*.csv'))
        if len(self.data_files) == 0:
            raise ValueError(f"No .csv files found in the folder: {data_folder}")

        # Determine the minimum number of frames across all CSV files
        self.min_frames = self.get_min_frames()

        self.data_list = []
        self.label_list = []
        self.load_data()

        if normalization:
            self.get_mean_map()

    def get_min_frames(self):
        min_frames = float('inf')
        for data_path in self.data_files:
            csv_data = pd.read_csv(data_path, header=None)
            num_frames = csv_data.shape[0]
            if num_frames < min_frames:
                min_frames = num_frames
        return min_frames

    def load_data(self):
        for data_path in self.data_files:
            # Extract Participant ID and Event ID from the filename
            filename = os.path.basename(data_path)
            match = re.match(r'Pt(\d+)_(C|PD)_n_(\d+)\.csv.*', filename)
            if not match:
                print(f"Skipping file with incorrect format: {filename}")
                continue

            participant_id = int(match.group(1))
            event_id = int(match.group(3))

            # Find the corresponding row in the label dataframe
            participant_filter = self.label_df['Participant ID number'] == participant_id
            event_filter = self.label_df['Turn ID'] == event_id
            row = self.label_df[participant_filter & event_filter]

            if row.empty:
                print(f"No matching label found for participant {participant_id} and event {event_id}, skipping.")
                continue

            pd_or_c = row['PD_or_C'].values[0]
            if pd_or_c == 'C':
                label = np.array([0])  # Healthy individual
            elif pd_or_c == 'PD':
                label = np.array([1])  # Patient with disease
            else:
                print(f"Unexpected value in PD_or_C column: {pd_or_c}, skipping.")
                continue

            # Load the data from the CSV file
            csv_data = pd.read_csv(data_path, header=None)

            # 数据集中没有无关列，直接限制帧数
            csv_data = csv_data.iloc[:self.min_frames, :].values

            # 检查列数是否正确（51列 = 17个关键点 × 3维）
            if csv_data.shape[1] != 51:
                print(f"Unexpected number of columns in {filename} (expected 51, got {csv_data.shape[1]}), skipping.")
                continue

            # 计算帧数和关键点数量
            T = csv_data.shape[0]  # 时间帧数
            V = 17  # 关键点数量（更新为 17）

            # 重塑数据为 (T, V, 3)
            data = csv_data.reshape((T, V, 3))

            # 将数据和标签添加到列表
            self.data_list.append(data)
            self.label_list.append(label)

        # 检查是否有有效数据
        if len(self.data_list) == 0:
            raise ValueError("No valid data files were loaded. Please check your dataset.")

        # 转换列表为 numpy 数组
        self.data = np.array(self.data_list)
        self.label = np.array(self.label_list)

    def get_mean_map(self):
        data = self.data
        N, T, V, C = data.shape
        self.mean_map = data.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).mean(axis=0)
        self.std_map = data.reshape((N * T * V, C)).std(axis=0).reshape((1, 1, V, C))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]  # 原始形状: (T, V, C)

        # 扩展为 4D: 增加一个额外的维度以匹配 (C, T, V, M)
        # 假设 M 表示帧中的人数，因为只有一个实例，所以 M = 1
        data_numpy = np.expand_dims(data_numpy, axis=-1)  # 形状变为 (T, V, C, 1)
        data_numpy = np.transpose(data_numpy, (2, 0, 1, 3))  # 重排顺序为 (C, T, V, M)

        label = self.label[index]

        # 计算有效帧的数量
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # 设置目标帧数为 509 帧
        target_frame_num = 100

        # 使用 auto_pading 函数将数据填充为 509 帧
        data_numpy = tools.auto_pading(data_numpy, target_frame_num, random_pad=False)

        # 调用 tools.valid_crop_resize 时，数据应是 4 维的
        p_interval = self.p_interval if isinstance(self.p_interval, (list, tuple)) else [self.p_interval]
        data_numpy = tools.valid_crop_resize(data_numpy, target_frame_num, p_interval, target_frame_num)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
