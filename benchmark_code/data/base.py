import os
import logging
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .mm_pre import MMDataset
from . import benchmarks

__all__ = ['DataManager']


class DataManager:

    def __init__(self, args, logger_name='Multimodal Intent Recognition'):

        self.logger = logging.getLogger(logger_name)

        self.benchmarks = benchmarks[args.dataset]

        self.data_path = os.path.join(args.data_path)
        # data_mode = 'emotion' or 'goal'
        if args.data_mode == 'emotion':
            self.label_list = self.benchmarks["emotion_labels"]
        elif args.data_mode == 'goal':
            self.label_list = self.benchmarks['goal_labels']
        else:
            raise ValueError('The input data mode is not supported.')
        self.logger.info('Lists of intent labels are: %s',
                         str(self.label_list))
        # self.id_split = self._get_id_split(args)
        args.num_labels = len(self.label_list)
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim, args.audio_image_dim = \
            self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims'][
                'video'], self.benchmarks['feat_dims']['image'], self.benchmarks['feat_dims']['audio']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = \
            self.benchmarks['max_seq_lengths']['text'], self.benchmarks[
                'max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']

        self.train_data_index, self.train_label_ids, self.train_modality_info = self._get_indexes_annotations(
            os.path.join(self.data_path, 'train_new_update.json'), args.data_mode, args)
        self.dev_data_index, self.dev_label_ids, self.dev_modality_info = self._get_indexes_annotations(
            os.path.join(self.data_path, 'dev_new_update.json'), args.data_mode, args)
        self.test_data_index, self.test_label_ids, self.test_modality_info = self._get_indexes_annotations(
            os.path.join(self.data_path, 'test_new_update.json'), args.data_mode, args)

        self.unimodal_feats = self._get_unimodal_feats(args, self._get_attrs())
        self.mm_data = self._get_multimodal_data(args)
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)

    def _get_indexes_annotations(self, json_path, data_mode, args):
        data_index = []
        label_ids = []
        support_label_ids = []
        # load train/dev/test json file
        with open(json_path, 'r') as f:
            datas = json.load(f)
        for data in datas:
            data_index_int = int(data['id'])
            data_index.append(data_index_int)
            support_label_ids.append(data['type'])
            if data_mode == 'emotion':
                emotion = torch.zeros(args.class_num_emotion)
                emotion[int(data['emotion'])] = 1
                label_ids.append(emotion)
                # label_ids.append(data['emotion'])
                ori_data_goal = data['goal']
                data_goal = np.zeros(21)
                data_goal[ori_data_goal] = 1
                # support_label_ids.append(data_goal)
            elif data_mode == 'goal':
                intent = torch.zeros(args.class_num_intent)
                goal = data['goal']
                for i in goal:
                    intent[i] = 1
                label_ids.append(intent)
                # label_ids.append(data['goal'])
                ori_data_emotion = data['emotion']
                data_emotion = np.zeros(11)
                data_emotion[ori_data_emotion] = 1
                # support_label_ids.append(data_emotion)
            else:
                raise ValueError('The input data mode is not supported.')
        return data_index, label_ids, support_label_ids
        # get indexes and annotations

    def _get_unimodal_feats(self, args, attrs):
        # text_feats = [text_feats_train, text_feats_dev, text_feats_test]
        # text_feats = TextDataset(args, attrs).feats
        # video_feats = VideoDataset(args, attrs).feats
        # audio_feats = AudioDataset(args, attrs).feats
        text_pkl_path = os.path.join(self.data_path, 'text_feats_final.npy')
        video_pkl_path = os.path.join(self.data_path, 'video_feats_final.npy')
        image_pkl_path = os.path.join(self.data_path, 'image_feats_final.npy')
        audio_pkl_path = os.path.join(self.data_path, 'audio_feats_final.npy')
        print(audio_pkl_path)
        # load pkl file
        text_feats = np.load(text_pkl_path, allow_pickle=True)
        video_feats = np.load(video_pkl_path, allow_pickle=True)
        image_feats = np.load(image_pkl_path, allow_pickle=True)
        audio_feats = np.load(audio_pkl_path, allow_pickle=True)

        # split these feats into train, dev, test according to indexes
        # if exists, load from pkl file, else use 0 to pad
        text_feats_train, text_feats_dev, text_feats_test = [], [], []
        video_feats_train, video_feats_dev, video_feats_test = [], [], []
        image_feats_train, image_feats_dev, image_feats_test = [], [], []
        audio_feats_train, audio_feats_dev, audio_feats_test = [], [], []

        for i in range(len(self.train_data_index)):
            text_feats_train.append(text_feats[self.train_data_index[i]])
            video_feats_train.append(video_feats[self.train_data_index[i]])
            image_feats_train.append(image_feats[self.train_data_index[i]])
            audio_feats_train.append(audio_feats[self.train_data_index[i]])

        for i in range(len(self.dev_data_index)):
            text_feats_dev.append(text_feats[self.dev_data_index[i]])
            video_feats_dev.append(video_feats[self.dev_data_index[i]])
            image_feats_dev.append(image_feats[self.dev_data_index[i]])
            audio_feats_dev.append(audio_feats[self.dev_data_index[i]])

        for i in range(len(self.test_data_index)):
            text_feats_test.append(text_feats[self.test_data_index[i]])
            video_feats_test.append(video_feats[self.test_data_index[i]])
            image_feats_test.append(image_feats[self.test_data_index[i]])
            audio_feats_test.append(audio_feats[self.test_data_index[i]])

        # element-wise sum the video and image features
        video_feats_train = np.array(video_feats_train)
        image_feats_train = np.array(image_feats_train)
        video_feats_dev = np.array(video_feats_dev)
        image_feats_dev = np.array(image_feats_dev)
        video_feats_test = np.array(video_feats_test)
        image_feats_test = np.array(image_feats_test)
        # extend the video dimension from 512 to 768
        video_feats_train = np.concatenate(
            (video_feats_train, np.zeros((video_feats_train.shape[0], 256))), axis=1)
        video_feats_dev = np.concatenate(
            (video_feats_dev, np.zeros((video_feats_dev.shape[0], 256))), axis=1)
        video_feats_test = np.concatenate(
            (video_feats_test, np.zeros((video_feats_test.shape[0], 256))), axis=1)

        video_feats_train = image_feats_train+video_feats_train
        video_feats_dev = image_feats_dev + video_feats_dev
        video_feats_test = image_feats_test + video_feats_test

        # return {
        #     'text': [text_feats_train, text_feats_dev, text_feats_test],
        #     'video': [video_feats_train, video_feats_dev, video_feats_test],
        #     'audio': [audio_feats_train, audio_feats_dev, audio_feats_test]
        # }
        text_feats = {}
        text_feats['train'] = text_feats_train
        text_feats['dev'] = text_feats_dev
        text_feats['test'] = text_feats_test

        video_feats = {}
        video_feats['train'] = video_feats_train
        video_feats['dev'] = video_feats_dev
        video_feats['test'] = video_feats_test

        audio_feats = {}
        audio_feats['train'] = audio_feats_train
        audio_feats['dev'] = audio_feats_dev
        audio_feats['test'] = audio_feats_test
        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats
        }

    def _get_multimodal_data(self, args):

        text_data = self.unimodal_feats['text']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']

        mm_train_data = MMDataset(
            self.train_label_ids, text_data['train'], video_data['train'], audio_data['train'], self.train_modality_info)
        mm_dev_data = MMDataset(
            self.dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], self.dev_modality_info)
        mm_test_data = MMDataset(
            self.test_label_ids, text_data['test'], video_data['test'], audio_data['test'], self.test_modality_info)

        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):

        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(
            data['train'], shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
        dev_dataloader = DataLoader(
            data['dev'], batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(
            data['test'], batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

        self.logger.info('Generate Dataloader Finished...')

        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }

    def _get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
