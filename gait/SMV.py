import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from torchsummary import summary


class CNN_FoG_trans_2022(nn.Module):
    def __init__(self):
        super(CNN_FoG_trans_2022, self).__init__()
        # Conv2d
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=77, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=77, out_channels=685, kernel_size=7, padding=3)

        self.conv3 = nn.Conv2d(in_channels=685, out_channels=128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=464, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(in_channels=464, out_channels=464, kernel_size=5, padding=2)

        self.conv7 = nn.Conv2d(in_channels=464, out_channels=101, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=101, out_channels=101, kernel_size=3, padding=1)
        # Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        # FC
        self.fc1 = nn.Linear(101 * 98 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        # BatchNorm
        self.norm1 = nn.BatchNorm2d(685)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(464)
        self.norm4 = nn.BatchNorm2d(101)
        # activation
        self.softsigh = nn.Softsign()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # dropout
        self.dropout = nn.Dropout(0.4)

        # first Conv2D Block [b, 1, 200, 20]->[b, 685, 200, 20]
        self.convBlock1 = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.pool1,
            self.norm1,
            self.softsigh,
            self.dropout
        )
        # Second Conv2D Block [b, 685, 200, 20]->[b, 128, 200, 20]
        self.convBlock2 = nn.Sequential(
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.norm2,
            self.softsigh,
            self.dropout
        )
        # Third Conv2D Block [b, 128, 200, 20]->[b, 464, 200, 20]
        self.convBlock3 = nn.Sequential(
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.norm3,
            self.softsigh,
            self.dropout
        )
        # Fourth Conv2D Block [b, 464, 200, 20]->[b, 101, 200, 20]
        self.convBlock4 = nn.Sequential(
            self.conv7,
            self.relu,
            self.conv8,
            self.relu,
            self.norm4,
            self.softsigh,
            self.dropout
        )
        # Output Layer [b, 101, 200, 20]->[b]
        self.output = nn.Sequential(
            nn.Flatten(),
            self.fc1,
            self.relu,
            self.fc2,
            self.sigmoid
        )

    def forward(self, x):
        """
        :param x: [b, 1, 200, 20]
        :return: [b]
        """
        # print("input:", x.shape)
        x = x.to(dtype=torch.float32)
        # first Conv2D Block [b, 1, 200, 20]->[b, 685, 200, 20]
        x = self.convBlock1(x)
        # print("after block1:", x.shape)
        # second Conv2D Block [b, 685, 98, 8]->[b, 128, 98, 8]
        x = self.convBlock2(x)
        # print("after block2:", x.shape)
        # third Conv2D Block [b, 128, 98, 8]->[b, 464, 98, 8]
        x = self.convBlock3(x)
        # print("after block3:", x.shape)
        # fourth Conv2D Block [b, 464, 98, 8]->[b, 101, 98, 8]
        x = self.convBlock4(x)
        # print("after block4:", x.shape)
        # output layer [b, 101, 98, 8]->[b]
        x = self.output(x)
        # print("output:", x.shape)
        return x


class CNN_FoG_simple(nn.Module):
    def __init__(self):
        super(CNN_FoG_simple, self).__init__()
        # Conv2d
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        # FC
        self.fc1 = nn.Linear(8 * 100 * 10, 128)
        self.fc2 = nn.Linear(128, 1)
        # BatchNorm
        self.norm_16 = nn.BatchNorm2d(16)
        self.norm_8 = nn.BatchNorm2d(8)
        # MaxPooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # dropout
        self.dropout = nn.Dropout(0.3)

        # first Conv2D Block [b, 6, 200, 20]->[b, 16, 202, 22]
        self.convBlock1 = nn.Sequential(
            self.conv1,
            self.norm_16,
            self.relu,
            self.pool1,
        )
        # Second Conv2D Block [b, 16, 202, 22]->[b, 32, 202, 22]
        self.convBlock2 = nn.Sequential(
            self.conv2,
            self.norm_16,
            self.relu,
            # self.pool2,
        )
        # Third Conv2D Block [b, 32, 202, 22]->[b, 16, 202, 22]
        self.convBlock3 = nn.Sequential(
            self.conv3,
            self.norm_8,
            self.relu,
            # self.pool3,
        )
        # Output Layer [b, 16, 202, 22]->[b]
        self.output = nn.Sequential(
            nn.Flatten(),
            self.fc1,
            # self.relu,
            self.fc2,
            self.dropout,
            self.sigmoid
        )

    def forward(self, x):
        """
        :param x: [b, 1, 200, 20]
        :return: [b]
        """
        # print("input:", x.shape)
        x = x.to(dtype=torch.float32)
        # first Conv2D Block [b, 1, 200, 20]->[b, 64, 200, 20]
        x = self.convBlock1(x)
        # print("after block1:", x.shape)
        # second Conv2D Block [b, 64, 98, 8]->[b, 64, 98, 8]
        x = self.convBlock2(x)
        # print("after block2:", x.shape)
        # third Conv2D Block [b, 64, 98, 8]->[b, 64, 98, 8]
        x = self.convBlock3(x)
        # print("after block3:", x.shape)
        x = self.output(x)
        # print("output:", x.shape)
        return x


class CNN_1D_Axis(nn.Module):
    def __init__(self):
        super(CNN_1D_Axis, self).__init__()
        # ConvBlocks for data in x axis
        self.convBlock_x_1_s = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_x_2_s = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_x_3_s = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_x_1_l = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_x_2_l = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_x_3_l = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=11, padding=5),
            nn.ReLU(),
        )

        # ConvBlocks for data in y axis
        self.convBlock_y_1_s = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_y_2_s = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_y_3_s = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_y_1_l = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_y_2_l = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_y_3_l = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=11, padding=5),
            nn.ReLU(),
        )

        # ConvBlocks for data in z axis
        self.convBlock_z_1_s = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_z_2_s = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_z_3_s = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convBlock_z_1_l = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_z_2_l = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=11, padding=5),
            nn.ReLU(),
        )
        self.convBlock_z_3_l = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=11, padding=5),
            nn.ReLU(),
        )

        self.att_x = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )
        self.att_y = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )
        self.att_z = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
        )
        self.att_axis = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 200, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(),
        )

        # self.att = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(240*200, 1),
        #     nn.Sigmoid(),
        # )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.pool = nn.MaxPool1d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 100, 128),
            nn.Linear(128, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )
        self.svm_x = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(128 * 100, 128),
            nn.Linear(128, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )
        self.svm_y = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(128 * 100, 128),
            nn.Linear(128, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )
        self.svm_z = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(128 * 100, 128),
            nn.Linear(128, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

        self.pool_svm = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        data = torch.transpose(x, 1, 2).reshape((-1, 3, 20, 200))
        data_x = data[:, 0, :, :]
        data_y = data[:, 1, :, :]
        data_z = data[:, 2, :, :]

        # DenseNet for x axis
        out_x_1_s = self.convBlock_x_1_s(data_x)
        # out_x_2_s = torch.concat((out_x_1_s, self.convBlock_x_2_s(out_x_1_s)), dim=1)
        # out_x_3_s = torch.concat((out_x_1_s, out_x_2_s, self.convBlock_x_3_s(out_x_2_s)), dim=1)
        out_x_1_l = self.convBlock_x_1_l(data_x)
        # out_x_2_l = torch.concat((out_x_1_l, self.convBlock_x_2_l(out_x_1_l)), dim=1)
        # out_x_3_l = torch.concat((out_x_1_l, out_x_2_l, self.convBlock_x_3_l(out_x_2_l)), dim=1)

        # DenseNet for y axis
        out_y_1_s = self.convBlock_y_1_s(data_y)
        # out_y_2_s = torch.concat((out_y_1_s, self.convBlock_y_2_s(out_y_1_s)), dim=1)
        # out_y_3_s = torch.concat((out_y_1_s, out_y_2_s, self.convBlock_y_3_s(out_y_2_s)), dim=1)
        out_y_1_l = self.convBlock_y_1_l(data_y)
        # out_y_2_l = torch.concat((out_y_1_l, self.convBlock_y_2_l(out_y_1_l)), dim=1)
        # out_y_3_l = torch.concat((out_y_1_l, out_y_2_l, self.convBlock_y_3_l(out_y_2_l)), dim=1)

        # DenseNet for z axis
        out_z_1_s = self.convBlock_z_1_s(data_z)
        # out_z_2_s = torch.concat((out_z_1_s, self.convBlock_z_2_s(out_z_1_s)), dim=1)
        # out_z_3_s = torch.concat((out_z_1_s, out_z_2_s, self.convBlock_z_3_s(out_z_2_s)), dim=1)
        out_z_1_l = self.convBlock_z_1_l(data_z)
        # out_z_2_l = torch.concat((out_z_1_l, self.convBlock_z_2_l(out_z_1_l)), dim=1)
        # out_z_3_l = torch.concat((out_z_1_l, out_z_2_l, self.convBlock_z_3_l(out_z_2_l)), dim=1)

        # Fusion for attribute
        # out_s = torch.concat((out_x_3_s, out_y_3_s, out_z_3_s), dim=1)
        # out_l = torch.concat((out_x_3_l, out_y_3_l, out_z_3_l), dim=1)
        # weight = self.sigmoid(self.att(out_s))
        # weight = weight.unsqueeze(1).repeat(1, 240, 200)
        # out_fused = out_s * weight + out_l * (1 - weight)
        # out_s = out_s * weight
        # out_l = out_l * (1 - weight)
        # out_fused = torch.concat((out_s, out_l), dim=1)

        # AFF Fusion for x axis
        out_x = torch.concat((out_x_1_l, out_x_1_s), dim=1)
        weight_s = self.sigmoid(self.att_x(out_x))
        out_x_s = out_x_1_s * weight_s
        out_x_l = out_x_1_l * (1 - weight_s)
        out_x_fused = torch.concat((out_x_s, out_x_l), dim=1)
        # out_x_fused = out_x_1_s * weight_s + out_x_1_l * (1 - weight_s)

        # AFF Fusion for y axis
        out_y = torch.concat((out_y_1_l, out_y_1_s), dim=1)
        weight_s = self.sigmoid(self.att_y(out_y))
        # out_y_fused = out_y_1_s * weight_s + out_y_1_l * (1 - weight_s)
        out_y_s = out_y_1_s * weight_s
        out_y_l = out_y_1_l * (1 - weight_s)
        out_y_fused = torch.concat((out_y_s, out_y_l), dim=1)

        # AFF Fusion for z axis
        out_z = torch.concat((out_z_1_l, out_z_1_s), dim=1)
        weight_s = self.sigmoid(self.att_z(out_z))
        # out_z_fused = out_z_1_s * weight_s + out_z_1_l * (1 - weight_s)
        out_z_s = out_z_1_s * weight_s
        out_z_l = out_z_1_l * (1 - weight_s)
        out_z_fused = torch.concat((out_z_s, out_z_l), dim=1)

        # AFF Fusion for three axis
        # out_xyz = torch.concat((out_x_fused, out_y_fused, out_z_fused), dim=1)
        # weight_xyz = self.att_axis(out_xyz).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 200)
        # weight_x = weight_xyz[:, 0, :, :]
        # weight_y = weight_xyz[:, 1, :, :]
        # weight_z = weight_xyz[:, 2, :, :]
        # out_x_fused *= weight_x
        # out_y_fused *= weight_y
        # out_z_fused *= weight_z
        # # out_xyz_fused = out_x_fused * weight_x + out_y_fused * weight_y + out_z_fused * weight_z
        # out_xyz_fused = torch.concat((out_x_fused, out_y_fused, out_z_fused), dim=1)
        # out_xyz_fused = self.pool(out_xyz_fused)

        # Concat
        # out_concat = torch.concat((out_x_3_s,
        #                            out_y_3_s,
        #                            out_z_3_s,
        #                            out_x_3_l,
        #                            out_y_3_l,
        #                            out_z_3_l), dim=1)
        # out_concat = torch.concat((out_x_fused, out_y_fused, out_z_fused), dim=1)

        # Initiate Classifier
        # out = self.pool(out_xyz_fused)
        # out = self.classifier(out_xyz_fused)

        # Fusion for label
        label_x = self.svm_x(out_x_fused)
        label_y = self.svm_x(out_y_fused)
        label_z = self.svm_x(out_z_fused)
        out_all = torch.concat((label_x, label_y, label_z), dim=1)
        out = self.pool_svm(out_all)

        return out


class CNN_1D_Position(nn.Module):
    def __init__(self):
        super(CNN_1D_Position, self).__init__()
        # ConvBlocks for 10 sensors
        for i in range(1, 11):
            setattr(self, f"convBlock_{i}_1_s", nn.Sequential(
                nn.Conv1d(6, 16, kernel_size=3, padding=1),
                nn.ReLU(),
            ))
            setattr(self, f"convBlock_{i}_1_l", nn.Sequential(
                nn.Conv1d(6, 16, kernel_size=11, padding=5),
                nn.ReLU(),
            ))

        # Attention for sensors
        self.att_sensors = nn.Sequential(
            nn.AvgPool1d(32 * 2 * 100),
            nn.Flatten(),
            nn.Linear(10 * 1, 10),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.pool = nn.MaxPool1d(2)
        self.svm = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(10 * 1600, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

        # Self Attention:
        self.values = nn.Linear(320, 320, bias=False)
        self.keys = nn.Linear(320, 320, bias=False)
        self.queries = nn.Linear(320, 320, bias=False)
        self.fc_out = nn.Linear(3200, 3200)
        # self.svm_x = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Linear(128 * 100, 128),
        #     nn.Linear(128, 1),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid(),
        # )

        # self.pool_svm = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        data = torch.transpose(x, 1, 2).reshape((-1, 10, 6, 200))
        data_dict = {}
        for i in range(1, 11):
            data_dict[f"data_{i}"] = data[:, i - 1, :, :]

        # DenseNet for sensors
        out_dict = {}
        for i in range(1, 11):
            exec(f"out_{i}_1_s = self.pool(self.convBlock_{i}_1_s(data_dict['data_{i}']))")
            exec(f"out_{i}_1_l = self.pool(self.convBlock_{i}_1_l(data_dict['data_{i}']))")
            exec(f"out_dict['out_{i}'] = torch.cat((out_{i}_1_s, out_{i}_1_l), dim=1)")

        # Attention for sensors
        out_list = []
        for i in range(1, 11):
            out_list.append(out_dict[f'out_{i}'])
        out_all = torch.stack(out_list, dim=1)
        out_all = out_all.view(out_all.size(0), out_all.size(1), -1)

        # Channel Attention
        # weight_sensor = self.att_sensors(out_all).unsqueeze(-1)
        # out_fused = out_all * weight_sensor

        # Channel SelfAttention
        batch_size = out_all.size(0)
        values = out_all.clone().reshape(batch_size, 10, 10, 320)
        keys = out_all.clone().reshape(batch_size, 10, 10, 320)
        queries = out_all.clone().reshape(batch_size, 10, 10, 320)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        score = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = nn.functional.softmax(score / (3200 ** (1 / 2)), dim=3)
        out_fused = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(batch_size, 10, 3200)
        out_fused = self.fc_out(out_fused)

        out = self.svm(out_fused)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        score = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(score / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, len_feature,
                 if_pool=False, pool_size=2,
                 if_1x1=False, out_channel_1x1=128,
                 if_attention=True):
        super(ConvBlock, self).__init__()
        self.if_attention = if_attention
        self.convBlock1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.LayerNorm([out_channels, len_feature]),
            nn.BatchNorm1d(out_channels),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5),
            nn.ReLU(),
            # nn.LayerNorm([out_channels, len_feature]),
            nn.BatchNorm1d(out_channels),
        )
        self.convBlock3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=21, padding=10),
            nn.ReLU(),
            # nn.LayerNorm([out_channels, len_feature]),
            nn.BatchNorm1d(out_channels),
        )
        self.convBlock4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=51, padding=25),
            nn.ReLU(),
            # nn.LayerNorm([out_channels, len_feature]),
            nn.BatchNorm1d(out_channels),
        )
        self.pool = nn.MaxPool1d(pool_size)

        if if_pool:
            self.convBlock1.add_module('pool', nn.MaxPool1d(pool_size))
            self.convBlock2.add_module('pool', nn.MaxPool1d(pool_size))
            self.convBlock3.add_module('pool', nn.MaxPool1d(pool_size))
            self.convBlock4.add_module('pool', nn.MaxPool1d(pool_size))

        if if_1x1:
            self.convBlock1.add_module('conv_1x1',
                                       nn.Conv1d(out_channels, out_channel_1x1, kernel_size=1))
            self.convBlock1.add_module('relu',
                                       nn.ReLU())
            self.convBlock2.add_module('conv_1x1',
                                       nn.Conv1d(out_channels, out_channel_1x1, kernel_size=1))
            self.convBlock2.add_module('relu',
                                       nn.ReLU())
            self.convBlock3.add_module('conv_1x1',
                                       nn.Conv1d(out_channels, out_channel_1x1, kernel_size=1))
            self.convBlock3.add_module('relu',
                                       nn.ReLU())
            self.convBlock4.add_module('conv_1x1',
                                       nn.Conv1d(out_channels, out_channel_1x1, kernel_size=1))
            self.convBlock4.add_module('relu',
                                       nn.ReLU())
        if if_1x1:
            self.kernel_att = KernelAtt(4, out_channel_1x1 * 100)
        else:
            self.kernel_att = KernelAtt(4, out_channels * 100)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        out_1 = self.convBlock1(x)
        out_2 = self.convBlock2(x)
        out_3 = self.convBlock3(x)
        out_4 = self.convBlock4(x)

        if self.if_attention:
            out = torch.concat((out_1.unsqueeze(1),
                                out_2.unsqueeze(1),
                                out_3.unsqueeze(1),
                                out_4.unsqueeze(1)), dim=1)
            out_flatten = out.reshape(batch_size, 4, -1)
            # # out = out.
            weight_kernels = self.kernel_att(out_flatten).unsqueeze(-1).unsqueeze(-1)
            out_weighted = out * weight_kernels
            output = torch.cat([out_weighted[:, i] for i in range(4)], dim=1)
        else:
            output = torch.concat((out_1, out_2, out_3, out_4), dim=1)
        return output


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, len_feature):
        super(ChannelAtt, self).__init__()
        self.in_channels = in_channels
        self.len_feature = len_feature
        self.rate_pool = self.len_feature // 10

        self.att = nn.Sequential(
            nn.AvgPool1d(self.rate_pool),
            nn.Flatten(),
            nn.Linear(self.in_channels * 10, self.in_channels),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.att(x)

        return out


class KernelAtt(nn.Module):
    def __init__(self, in_channels, len_feature):
        super(KernelAtt, self).__init__()
        self.in_channels = in_channels
        self.len_feature = len_feature
        self.rate_pool = self.len_feature // 10

        self.att = nn.Sequential(
            nn.AvgPool1d(self.rate_pool),
            nn.Flatten(),
            nn.Linear(self.in_channels * 10, self.in_channels),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.att(x)

        return out


class FeatureAtt(nn.Module):
    def __init__(self, in_channels, len_feature):
        super(FeatureAtt, self).__init__()
        self.in_channels = in_channels
        self.len_feature = len_feature

        self.conv1d = nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=1)

        self.att = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.len_feature, self.len_feature),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv1d(x)
        out = self.att(out)

        return out


class DenseNet_2layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNet_2layer, self).__init__()
        self.inner_channel = 64
        self.in_channels_block2 = self.inner_channel * 4
        self.out_channels_1x1 = 128
        # self.in_channels_block3 = self.in_channels_block2 + out_channels * 3
        # self.convBlock1 = ConvBlock(in_channels=in_channels, out_channels=out_channels,
        #                             if_pool=True, pool_size=2)
        self.convBlock1 = ConvBlock(in_channels=in_channels,
                                    out_channels=self.inner_channel,
                                    len_feature=200,
                                    if_pool=True,
                                    pool_size=2)
        self.convBlock2 = ConvBlock(in_channels=self.in_channels_block2,
                                    out_channels=256,
                                    len_feature=100,
                                    if_1x1=True,
                                    out_channel_1x1=self.out_channels_1x1)
        self.att_channel_1 = ChannelAtt(self.in_channels_block2, len_feature=100)
        self.att_channel_2 = ChannelAtt(self.out_channels_1x1 * 4, len_feature=100)

    def forward(self, x):
        out_1 = self.convBlock1(x)

        weight_block1 = self.att_channel_1(out_1).unsqueeze(-1)
        out_1 = weight_block1 * out_1

        out_2 = self.convBlock2(out_1)

        weight_block2 = self.att_channel_2(out_2).unsqueeze(-1)
        out_2 = weight_block2 * out_2

        # out_3 = self.convBlock3(out_2)
        out = torch.concat((out_1, out_2), dim=1)
        # out = self.convBlock3(out)

        return out


class DenseNet_2layer_seperated(nn.Module):
    def __init__(self, in_channels, out_channels, if_attention=True):
        super(DenseNet_2layer_seperated, self).__init__()
        self.inner_channel = 16
        self.in_channels_block2 = self.inner_channel * 4
        self.out_channels_1x1 = 64
        # self.in_channels_block3 = self.in_channels_block2 + out_channels * 3
        # self.convBlock1 = ConvBlock(in_channels=in_channels, out_channels=out_channels,
        #                             if_pool=True, pool_size=2)
        if if_attention:
            self.convBlock1 = ConvBlock(in_channels=in_channels,
                                        out_channels=self.inner_channel,
                                        len_feature=200,
                                        if_pool=True,
                                        pool_size=2)
            self.convBlock2 = ConvBlock(in_channels=self.in_channels_block2,
                                        out_channels=64,
                                        len_feature=100,
                                        if_1x1=True,
                                        out_channel_1x1=self.out_channels_1x1)
        else:
            self.convBlock1 = ConvBlock(in_channels=in_channels,
                                        out_channels=self.inner_channel,
                                        len_feature=200,
                                        if_pool=True,
                                        pool_size=2,
                                        if_attention=False)
            self.convBlock2 = ConvBlock(in_channels=self.in_channels_block2,
                                        out_channels=64,
                                        len_feature=100,
                                        if_1x1=True,
                                        out_channel_1x1=self.out_channels_1x1,
                                        if_attention=False)
        self.att_channel_1 = ChannelAtt(self.in_channels_block2, len_feature=100)
        self.att_channel_2 = ChannelAtt(self.out_channels_1x1 * 4, len_feature=100)

    def forward(self, x):
        out_1 = self.convBlock1(x)

        weight_block1 = self.att_channel_1(out_1).unsqueeze(-1)
        out_1 = weight_block1 * out_1

        out_2 = self.convBlock2(out_1)

        weight_block2 = self.att_channel_2(out_2).unsqueeze(-1)
        out_2 = weight_block2 * out_2

        # out_3 = self.convBlock3(out_2)
        out = torch.concat((out_1, out_2), dim=1)
        # out = self.convBlock3(out)

        return out


class CNN_1D(nn.Module):
    def __init__(self, number_sensor=10):
        super(CNN_1D, self).__init__()
        # ConvBlocks
        # self.denseBlock_acc = DenseNet_2layer(in_channels=30, out_channels=32)
        # self.denseBlock_gyr = DenseNet_2layer(in_channels=30, out_channels=32)
        self.in_channels = number_sensor * 6
        self.denseBlock = DenseNet_2layer(in_channels=self.in_channels, out_channels=128)
        self.denseBlock_out1 = 64 * 4
        self.denseBlock_out2 = self.denseBlock_out1 + 128 * 4
        # self.denseBlock_out3 = self.denseBlock_out2 + 128 * 3
        # self.denseBlock_out = self.denseBlock_out1 + self.denseBlock_out2 + self.denseBlock_out3

        # Attention
        # self.att_feature = FeatureAtt(in_channels=self.denseBlock_out2, len_feature=100)
        # self.att_channel = ChannelAtt(in_channels=self.denseBlock_out2, len_feature=100)
        # self.att_acc = ChannelAtt(in_channels=32 * 12, len_feature=100)
        # self.att_gyr = ChannelAtt(in_channels=32 * 12, len_feature=100)
        # self.att_sensor = nn.Parameter(torch.tensor(0.5))

        # cross Attention:
        # self.att_cross_acc = SelfAttention(embed_size=100, heads=10)

        # self.svm_acc = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Linear((32 * 12) * 50, 128),
        #     nn.Linear(128, 1),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid(),
        # )
        # self.svm_gyr = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Linear((32 * 12) * 50, 128),
        #     nn.Linear(128, 1),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid(),
        # )
        self.svm = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(self.denseBlock_out2 * 50, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

        # self.pool_svm = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        data = torch.transpose(x, 1, 2)
        out_all = self.denseBlock(data)
        # data = data.reshape(-1, 10, 6, 200)
        # data = data.reshape(-1, 10, 2, 3, 200)
        # data_acc = data[:, :, 0, :, :].reshape(-1, 30, 200)
        # data_gyr = data[:, :, 1, :, :].reshape(-1, 30, 200)

        # weight_channel = self.att_channel(out_all).unsqueeze(-1)
        # out_all_weighted = weight_channel * out_all

        # weights_acc = self.att_acc(out_acc).unsqueeze(-1)
        # out_acc_fused = out_acc * weights_acc
        #
        # weights_gyr = self.att_gyr(out_gyr).unsqueeze(-1)
        # out_gyr_fused = out_gyr * weights_gyr

        # weights_feature = self.att_feature(out_all).unsqueeze(-2)
        # out_all_weighted = weights_feature * out_all
        out = self.svm(out_all)

        # label_acc = self.svm_acc(out_acc_fused)
        # label_gyr = self.svm_gyr(out_gyr_fused)
        #
        # weight_sensor = self.att_sensor.unsqueeze(0)
        # out = label_acc * weight_sensor + label_gyr * (1 - weight_sensor)

        return out


class SMV(nn.Module):
    def __init__(self, if_attention=True):
        super(SMV, self).__init__()
        # ConvBlocks
        '''
        input:[batch, 200, 60]
        output:[batch, 10, 320, 100]
        '''
        if if_attention:
            self.denseBlock = DenseNet_2layer_seperated(in_channels=6, out_channels=32)
        else:
            self.denseBlock = DenseNet_2layer_seperated(in_channels=6, out_channels=32, if_attention=False)
        self.denseBlock_out1 = 64 * 4
        self.denseBlock_out2 = self.denseBlock_out1 + 128 * 4

    def forward(self, x):
        data = torch.transpose(x, 1, 2)
        data = data.reshape(-1, 10, 6, 200)
        out = []
        for sensor in range(10):
            data_sensor = data[:, sensor, :, :]
            out.append(self.denseBlock(data_sensor))  # out.shape:[10, batch, 320, 100]
        out = torch.transpose(torch.stack(out), 0, 1)

        return out  # [batch, 10, 320, 100]


class F2NMP(nn.Module):
    def __init__(self, num_heads=8):
        super(F2NMP, self).__init__()
        self.num_sensors = 10
        self.feature_dim1 = 320
        self.feature_dim2 = 100
        self.flatten_dim = self.feature_dim1 * self.feature_dim2
        self.d_model = 128
        self.dropout = 0.1
        self.flatten_dim_context = self.d_model * (self.num_sensors - 1)

        self.pre_map_x = nn.Linear(self.flatten_dim, self.d_model)
        self.pre_map_context = nn.Linear(self.flatten_dim_context, self.d_model)
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.attn = nn.MultiheadAttention(embed_dim=self.d_model,
                                          num_heads=num_heads,
                                          dropout=self.dropout,
                                          batch_first=True)

        self.post_map_context = nn.Linear(self.d_model, self.flatten_dim)
        self.post_norm = nn.LayerNorm(self.flatten_dim)

        self.gate_x = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(self.num_sensors, self.num_sensors),
            nn.Sigmoid()
        )
        self.gate_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(self.num_sensors, self.num_sensors),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(num_features=self.num_sensors)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        '''
        input:[batch, 10, 320, 100]
        output:[batch, 10, 320, 100]
        '''
        batch_size = x.shape[0]
        num_sensors = self.num_sensors
        feature_dim1 = self.feature_dim1
        feature_dim2 = self.feature_dim2

        map_x = self.pre_map_x(x.view(batch_size, num_sensors, -1))

        current_feature = map_x.unsqueeze(2)
        flat_current_feature = current_feature.view(batch_size, num_sensors, -1)

        context = map_x.unsqueeze(2)
        context = context.repeat(1, 1, 10, 1)
        indices = torch.arange(10).repeat(batch_size, 1).to(x.device)
        mask = (indices.unsqueeze(1) != indices.unsqueeze(2))
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.d_model)
        context = context.masked_select(mask).view(batch_size, 10, 9, self.d_model)
        flat_context = context.view(batch_size, num_sensors, -1)
        map_context = self.pre_map_context(flat_context)

        attn_output, _ = self.attn(flat_current_feature,
                                   map_context,
                                   map_context,
                                   need_weights=False)
        attn_output = self.post_map_context(attn_output)
        attn_output = self.post_norm(attn_output)
        attn_output = self.relu(attn_output)
        fused_context = attn_output.view(batch_size, num_sensors, feature_dim1, feature_dim2)

        gate_x = self.gate_x(map_x)
        gate_x = gate_x.view(batch_size, num_sensors, 1, 1)
        gate_context = self.gate_context(attn_output)
        gate_context = gate_context.view(batch_size, num_sensors, 1, 1)

        fused = gate_x * x + gate_context * fused_context
        result = self.bn(fused)

        return result


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q.transpose(1, 2), K)
        attn_probs = self.softmax(attn_scores)

        out = torch.matmul(V, attn_probs.transpose(1, 2))
        return out


class N2NMP(nn.Module):
    def __init__(self):
        super(N2NMP, self).__init__()
        # MLP for ARM
        '''
        input:[batch, 2, 320, 100]
        output:[batch, 320*100*3]
        '''
        self.SA_ARM = SelfAttention(in_channels=320 * 2, out_channels=640)

        # MLP for THIGH
        '''
        input:[batch, 4, 320, 100]
        output:[batch, 320*100*3]
        '''
        self.SA_THIGH = SelfAttention(in_channels=320 * 4, out_channels=640)

        # MLP for FOOT
        '''
        input:[batch, 2, 320, 100]
        output:[batch, 320*100*3]
        '''
        self.SA_FOOT = SelfAttention(in_channels=320 * 2, out_channels=640)

        # MLP for TRUNK
        '''
        input:[batch, 2, 320, 200]
        output:[batch, 320*100*3]
        '''
        self.SA_TRUNK = SelfAttention(in_channels=320 * 2, out_channels=640)

    def forward(self, x):
        batch_size = x.shape[0]
        x_arm = x[:, -2:, :, :].view(batch_size, -1, 100)
        x_thigh = x[:, 2:6, :, :].view(batch_size, -1, 100)
        x_foot = x[:, 6:8, :, :].view(batch_size, -1, 100)
        x_trunk = x[:, 0:2, :, :].view(batch_size, -1, 100)

        fused_arm = self.SA_ARM(x_arm)
        fused_thigh = self.SA_THIGH(x_thigh)
        fused_foot = self.SA_FOOT(x_foot)
        fused_trunk = self.SA_TRUNK(x_trunk)

        return fused_arm, fused_thigh, fused_foot, fused_trunk


class Gate_Joint(nn.Module):
    def __init__(self, num_heads=8):
        super(Gate_Joint, self).__init__()

        self.MLP_MAP = nn.Sequential(
            nn.Conv1d(in_channels=640 * 3, out_channels=640, kernel_size=1),
            nn.ReLU(),
            # nn.BatchNorm1d(768)
        )

        self.attn = nn.MultiheadAttention(embed_dim=640,
                                          num_heads=num_heads)

        self.MLP_gate_main = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=640, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        self.MLP_gate_related = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=640, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, main_feature, related_features):
        '''
        :param main_feature: [batch, 320, 100]
        :param related_features: [batch, 3 * 320, 100]
        :return: [batch, 320, 100]
        '''

        batch_size, _, seq_length = main_feature.size()

        related_features_mapped = self.MLP_MAP(related_features)

        main_feature = main_feature.permute(2, 0, 1)
        related_features_mapped = related_features_mapped.permute(2, 0, 1)

        attn_out, _ = self.attn(main_feature, related_features_mapped, related_features_mapped)

        attn_out = attn_out.permute(1, 2, 0)
        main_feature = main_feature.permute(1, 2, 0)

        gate_main = self.MLP_gate_main(main_feature)
        gate_main = gate_main.view(batch_size, 1, 1)

        gate_related = self.MLP_gate_related(attn_out)
        gate_related = gate_related.view(batch_size, 1, 1)

        main_feature_gated = gate_main * main_feature
        related_features_gated = gate_related * attn_out

        updated_feature = main_feature_gated + related_features_gated

        return updated_feature


class IndependentMLP(nn.Module):
    def __init__(self):
        super(IndependentMLP, self).__init__()
        self.avg_pool = nn.AvgPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640 * 50, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.dropout2 = nn.Dropout(0.3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        hidden_state = self.fc1(x)
        hidden_state = self.dropout1(hidden_state)
        activated_hidden = self.relu(hidden_state)
        output = self.fc2(activated_hidden)
        output = self.dropout2(output)
        return output, hidden_state


class pre_IndependentMLP(nn.Module):
    def __init__(self):
        super(pre_IndependentMLP, self).__init__()
        self.avg_pool = nn.AvgPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640 * 50, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)
        self.dropout2 = nn.Dropout(0.3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        hidden_state = self.fc1(x)
        hidden_state = self.dropout1(hidden_state)
        activated_hidden = self.relu(hidden_state)
        output = self.fc2(activated_hidden)
        output = self.dropout2(output)
        return output, hidden_state


class MLP_OUTPUT(nn.Module):
    def __init__(self, num_outputs=4):
        super(MLP_OUTPUT, self).__init__()
        self.num_outputs = num_outputs
        self.mlps = nn.ModuleList([IndependentMLP() for _ in range(num_outputs)])

    def forward(self, x):
        '''
        :param x: [batch, 4, 600, 100]
        :return: [batch, 4]
        '''
        batch_size, num_outputs, channels, seq_length = x.size()
        channel_data = x.unbind(dim=1)
        final_outputs = []
        hidden_states = []

        for i in range(num_outputs):
            output, hidden_state = self.mlps[i](channel_data[i])
            final_outputs.append(output)
            hidden_states.append(hidden_state)

        final_outputs = torch.stack(final_outputs, dim=0)
        hidden_states = torch.stack(hidden_states, dim=0)

        return final_outputs, hidden_states


class pre_MLP_OUTPUT(nn.Module):
    def __init__(self, num_outputs=4):
        super(pre_MLP_OUTPUT, self).__init__()
        self.num_outputs = num_outputs
        self.mlps = nn.ModuleList([pre_IndependentMLP() for _ in range(num_outputs)])

    def forward(self, x):
        '''
        :param x: [batch, 4, 600, 100]
        :return: [batch, 4, 3]
        '''
        batch_size, num_outputs, channels, seq_length = x.size()
        channel_data = x.unbind(dim=1)
        final_outputs = []
        hidden_states = []

        for i in range(num_outputs):
            output, hidden_state = self.mlps[i](channel_data[i])
            final_outputs.append(output)
            hidden_states.append(hidden_state)

        final_outputs = torch.stack(final_outputs, dim=0)
        hidden_states = torch.stack(hidden_states, dim=0)

        return final_outputs, hidden_states


class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, node_features, adj_matrix):
        '''

        :param node_features: [batch, num_nodes, feature_size]
        :param adj_matrix: [batch, num_nodes, num_nodes]
        :return:
        '''
        # message passing
        aggregated_message = torch.matmul(adj_matrix, node_features)
        updated_features = self.mlp(aggregated_message + node_features)
        return updated_features


class AdjMatrixGenerator(nn.Module):
    def __init__(self, feature_size):
        super(AdjMatrixGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, hidden_state):
        '''
        :param hidden_state: [batch, num_nodes, feature_size]
        :return: adj_matrix: [batch, num_nodes, num_nodes]
        '''
        batch_size, num_nodes, feature_size = hidden_state.shape
        adj_matrix = torch.ones(batch_size, num_nodes, num_nodes).cuda()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    combined_feature = torch.cat([hidden_state[:, i, :], hidden_state[:, j, :]], dim=1)
                    adj_matrix[:, i, j] = self.mlp(combined_feature).squeeze(-1)

        return adj_matrix


class GNNModel(nn.Module):
    def __init__(self, num_classifiers, feature_size=128, hidden_size=128):
        super(GNNModel, self).__init__()
        self.num_classifiers = num_classifiers
        self.feature_size = feature_size

        self.gnn = GNNLayer(feature_size, hidden_size)
        self.adj_matrix_generator = AdjMatrixGenerator(feature_size + 1)
        self.global_decision_layer = nn.Sequential(
            nn.Linear(hidden_size * num_classifiers, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Dropout(0.3)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, logits_list, hidden_state_list):
        '''
        :param logits_list: [num_classifiers, batch_size, 1]
        :param hidden_state_list: [num_classifiers, batch_size, 128]
        :param adj_matrix: [num_classifiers, num_classifiers]
        :return: final label: [batch, 1]
        '''
        # 1. node construction [num_classifiers, batch_size, 128]
        node_features = []
        for i in range(self.num_classifiers):
            hidden_state = hidden_state_list[i]
            node_features.append(hidden_state)
        node_features = torch.stack(node_features, dim=0).cuda()
        batch_size = node_features.size(1)
        node_features = node_features.view(self.num_classifiers, batch_size, -1)
        node_features = node_features.permute(1, 0, 2)

        # 2. adjacency matrix generation
        adj_matrix = self.adj_matrix_generator(node_features)
        # adj_matrix = torch.ones(batch_size, 4, 4).cuda()

        # 3. message passing
        updated_node_features = self.gnn(node_features, adj_matrix)

        # 4. global decision
        updated_node_features = updated_node_features.view(batch_size, -1)
        output = self.global_decision_layer(updated_node_features)

        return output


class pre_GNNModel(nn.Module):
    def __init__(self, num_classifiers, feature_size=64, hidden_size=64):
        super(pre_GNNModel, self).__init__()
        self.num_classifiers = num_classifiers
        self.feature_size = feature_size

        self.gnn = GNNLayer(feature_size, hidden_size)
        self.adj_matrix_generator = AdjMatrixGenerator(feature_size + 1)
        self.global_decision_layer = nn.Sequential(
            nn.Linear(hidden_size * num_classifiers, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Dropout(0.3)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, logits_list, hidden_state_list):
        '''
        :param logits_list: [num_classifiers, batch_size, 1]
        :param hidden_state_list: [num_classifiers, batch_size, 128]
        :param adj_matrix: [num_classifiers, num_classifiers]
        :return: final label: [batch, 1]
        '''
        # 1. node construction [num_classifiers, batch_size, 128]
        node_features = []
        for i in range(self.num_classifiers):
            hidden_state = hidden_state_list[i]
            node_features.append(hidden_state)
        node_features = torch.stack(node_features, dim=0).cuda()
        batch_size = node_features.size(1)
        node_features = node_features.view(self.num_classifiers, batch_size, -1)
        node_features = node_features.permute(1, 0, 2)

        # 2. adjacency matrix generation
        adj_matrix = self.adj_matrix_generator(node_features)
        # adj_matrix = torch.ones(batch_size, 4, 4).cuda()

        # 3. message passing
        updated_node_features = self.gnn(node_features, adj_matrix)

        # 4. global decision
        updated_node_features = updated_node_features.view(batch_size, -1)
        output = self.global_decision_layer(updated_node_features)

        return output


class CO_FOG(nn.Module):
    def __init__(self):
        super(CO_FOG, self).__init__()
        # SMV
        '''
        input:[batch, 200, 60]
        output:[batch, 10, 320, 100]
        '''
        self.SMV = SMV(if_attention=True)

        # F2NMP
        '''
        input:[batch, 10, 320, 100]
        output:[batch, 10, 320, 100]
        '''
        self.F2NMP = F2NMP(num_heads=16)

        # N2NMP
        '''
        input:[batch, 10, 320, 100]
        output:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        '''
        self.N2Nmp = N2NMP()

        # Complementary Gate Joint
        '''
        input:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        output:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        '''
        self.Gate_Joints = nn.ModuleList([
            Gate_Joint(num_heads=16),  # For ARM
            Gate_Joint(num_heads=16),  # For THIGH
            Gate_Joint(num_heads=16),  # For FOOT
            Gate_Joint(num_heads=16)  # For TRUNK
        ])

        # JOINT OUTPUT
        '''
        input:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        output:[batch, 4], [batch, 4, 128]
        '''
        self.MLPs_OUTPUT = MLP_OUTPUT(num_outputs=4)

        # Cooprated Classification Agents
        '''
        input:[batch, 4], [batch, 4, 128]
        output:[batch, 1]
        '''
        self.GNN = GNNModel(num_classifiers=4)

    def forward(self, x):
        feature_smv = self.SMV(x)
        feature_nodes = self.F2NMP(feature_smv)
        feature_arm, feature_thigh, feature_foot, feature_trunk = self.N2Nmp(feature_nodes)
        features = torch.stack([feature_arm, feature_thigh, feature_foot, feature_trunk], dim=1)

        batch_size, num_nodes, feature_channels, feature_length = features.size()
        features_i = features.unsqueeze(2)  # [batch, 4, 1, 768, 200]
        features_j = features.unsqueeze(1)  # [batch, 1, 4, 768, 200]
        diff = torch.abs(features_i - features_j)  # [batch, 4, 4, 768, 200]
        mask = torch.eye(num_nodes, dtype=torch.bool, device=feature_nodes.device).unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1)
        diff = diff.masked_fill(mask, 0)
        diff = diff.masked_select(~mask).view(batch_size, num_nodes, num_nodes - 1, feature_channels, feature_length)
        related_features = diff.view(batch_size, num_nodes, (num_nodes - 1) * feature_channels, feature_length)

        updated_features = torch.stack([
            self.Gate_Joints[i](features[:, i, :, :], related_features[:, i, :, :])
            for i in range(4)
        ], dim=1)

        logits, hidden_states = self.MLPs_OUTPUT(updated_features)

        logit_final = self.GNN(logits, hidden_states)

        # final_label = torch.mean(labels, dim=1).unsqueeze(-1)
        # final_label = labels.max(dim=1, keepdim=True)[0]
        # final_labels = torch.cat([final_label, labels], dim=1)

        return logits, logit_final


class pre_CO_FOG(nn.Module):
    def __init__(self):
        super(pre_CO_FOG, self).__init__()
        # SMV
        '''
        input:[batch, 200, 60]
        output:[batch, 10, 320, 100]
        '''
        self.SMV = SMV(if_attention=True)

        # F2NMP
        '''
        input:[batch, 10, 320, 100]
        output:[batch, 10, 320, 100]
        '''
        self.F2NMP = F2NMP(num_heads=8)

        # N2NMP
        '''
        input:[batch, 10, 320, 100]
        output:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        '''
        self.N2Nmp = N2NMP()

        # Complementary Gate Joint
        '''
        input:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        output:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        '''
        self.Gate_Joints = nn.ModuleList([
            Gate_Joint(num_heads=8),  # For ARM
            Gate_Joint(num_heads=8),  # For THIGH
            Gate_Joint(num_heads=8),  # For FOOT
            Gate_Joint(num_heads=8)  # For TRUNK
        ])

        # JOINT OUTPUT
        '''
        input:[batch, 640, 100] * 4 (arm, thigh, foot, trunk)
        output:[batch, 4], [batch, 4, 128]
        '''
        self.MLPs_OUTPUT = pre_MLP_OUTPUT(num_outputs=4)

        # Cooprated Classification Agents
        '''
        input:[batch, 4], [batch, 4, 128]
        output:[batch, 1]
        '''
        self.GNN = pre_GNNModel(num_classifiers=4)

    def forward(self, x):
        feature_smv = self.SMV(x)
        feature_nodes = self.F2NMP(feature_smv)
        feature_arm, feature_thigh, feature_foot, feature_trunk = self.N2Nmp(feature_nodes)
        features = torch.stack([feature_arm, feature_thigh, feature_foot, feature_trunk], dim=1)

        batch_size, num_nodes, feature_channels, feature_length = features.size()
        features_i = features.unsqueeze(2)  # [batch, 4, 1, 768, 200]
        features_j = features.unsqueeze(1)  # [batch, 1, 4, 768, 200]
        diff = torch.abs(features_i - features_j)  # [batch, 4, 4, 768, 200]
        mask = torch.eye(num_nodes, dtype=torch.bool, device=feature_nodes.device).unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1)
        diff = diff.masked_fill(mask, 0)
        diff = diff.masked_select(~mask).view(batch_size, num_nodes, num_nodes - 1, feature_channels, feature_length)
        related_features = diff.view(batch_size, num_nodes, (num_nodes - 1) * feature_channels, feature_length)

        updated_features = torch.stack([
            self.Gate_Joints[i](features[:, i, :, :], related_features[:, i, :, :])
            for i in range(4)
        ], dim=1)

        logits, hidden_states = self.MLPs_OUTPUT(updated_features)

        logit_final = self.GNN(logits, hidden_states)

        # final_label = torch.mean(labels, dim=1).unsqueeze(-1)
        # final_label = labels.max(dim=1, keepdim=True)[0]
        # final_labels = torch.cat([final_label, labels], dim=1)

        return logits, logit_final


class pre_CNN_1D(nn.Module):
    def __init__(self):
        super(pre_CNN_1D, self).__init__()
        # ConvBlocks
        # self.denseBlock_acc = DenseNet_2layer(in_channels=30, out_channels=32)
        # self.denseBlock_gyr = DenseNet_2layer(in_channels=30, out_channels=32)
        self.denseBlock = DenseNet_2layer(in_channels=60, out_channels=128)
        self.denseBlock_out1 = 64 * 4
        self.denseBlock_out2 = self.denseBlock_out1 + 128 * 4
        # self.denseBlock_out3 = self.denseBlock_out2 + 128 * 3
        # self.denseBlock_out = self.denseBlock_out1 + self.denseBlock_out2 + self.denseBlock_out3

        # Attention
        # self.att_feature = FeatureAtt(in_channels=self.denseBlock_out2, len_feature=100)
        # self.att_channel = ChannelAtt(in_channels=self.denseBlock_out2, len_feature=100)
        # self.att_acc = ChannelAtt(in_channels=32 * 12, len_feature=100)
        # self.att_gyr = ChannelAtt(in_channels=32 * 12, len_feature=100)
        # self.att_sensor = nn.Parameter(torch.tensor(0.5))

        # cross Attention:
        # self.att_cross_acc = SelfAttention(embed_size=100, heads=10)

        # self.svm_acc = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Linear((32 * 12) * 50, 128),
        #     nn.Linear(128, 1),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid(),
        # )
        # self.svm_gyr = nn.Sequential(
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Linear((32 * 12) * 50, 128),
        #     nn.Linear(128, 1),
        #     nn.Dropout(0.3),
        #     nn.Sigmoid(),
        # )
        self.svm = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(self.denseBlock_out2 * 50, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Dropout(0.3),
        )
        self.Softmax = nn.Softmax(dim=1)

        # self.pool_svm = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        data = torch.transpose(x, 1, 2)
        out_all = self.denseBlock(data)
        out = self.svm(out_all)
        out = self.Softmax(out)

        return out


if __name__ == '__main__':
    # 创建模型实例
    # model = pre_CNN_1D().cuda()
    # model = F2NMP()
    # model = N2NMP()
    # model = CO_FOG()
    # model = pre_CO_FOG().cuda()
    model = CNN_1D()

    # # 打印模型结构
    model.cuda()
    print(model)
    # summary(model, (200, 60, ))

    # 测试维度
    # numpy_input = np.random.randn(64, 200, 60)
    # torch_input = torch.tensor(numpy_input, dtype=torch.float32).cuda()
    # output = model(torch_input)

    # 性能测试
    # model = model.to('cuda')
    # numpy_input = np.random.randn(64, 200, 60)
    # torch_input = torch.tensor(numpy_input, dtype=torch.float32).to('cuda')
    #
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     with record_function('model_inference'):
    #         with torch.no_grad():
    #             output_labels, output_scores = model(torch_input)
    #
    # print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))

    # prof.export_chrome_trace("trace.json")
