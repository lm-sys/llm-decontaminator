from enum import auto, IntEnum
import os

import numpy as np


INPUT_IDS_DTYPE = np.uint16
TMP_PADDING_VALUE = np.iinfo(INPUT_IDS_DTYPE).max


class DatasetMode(IntEnum):
    """Separator styles."""

    ONE_D = auto()
    TWO_D = auto()


class MmapDataset:
    def __init__(self, token_ids, mask, index):
        self.token_ids = token_ids
        self.mask = mask
        self.index = index

        if len(self.token_ids.shape) == 1:
            self.mode = DatasetMode.ONE_D
        else:
            self.mode = DatasetMode.TWO_D

    @staticmethod
    def load(filename: str):
        token_ids = np.lib.format.open_memmap(filename)

        mask_filename = MmapDataset.token_filename_to_mask_filename(filename)
        if os.path.exists(mask_filename):
            mask = np.lib.format.open_memmap(mask_filename)
            assert mask.shape == token_ids.shape
        else:
            mask = None  # Assume no padding and no ignore

        index_filename = MmapDataset.token_filename_to_index_filename(filename)
        index = np.lib.format.open_memmap(index_filename)

        return MmapDataset(token_ids, mask, index)

    def save(self, filename: str):
        mask_filename = self.token_filename_to_mask_filename(filename)
        index_filename = self.token_filename_to_index_filename(filename)
        with open(filename, "wb") as fout:
            np.save(fout, self.token_ids.astype(INPUT_IDS_DTYPE))
        with open(mask_filename, "wb") as fout:
            np.save(fout, self.mask.astype(np.int8))
        with open(index_filename, "wb") as fout:
            np.save(fout, self.index.astype(np.int64))

    def __len__(self):
        return len(self.index) - 1

    def __getitem__(self, i):
        if self.mode == DatasetMode.TWO_D:
            i = i
        else:
            raise NotImplementedError()
            #beg, end = self.index[i], self.index[i+1]
            #token_ids = self.token_ids[beg:end]

        token_ids = self.token_ids[i]
        if self.mask is not None:
            mask = self.mask[i]
            padding_mask = np.bitwise_and(mask, 0x1).astype(bool)
            is_ignore = np.bitwise_and(np.right_shift(mask, 1), 0x1).astype(bool)
        else:
            padding_mask = np.ones(token_ids.shape, dtype=bool)
            is_ignore = np.zeros(token_ids.shape, dtype=bool)

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "is_ignore": is_ignore,
        }

    @staticmethod
    def token_filename_to_mask_filename(filename):
        assert ".tok" in filename
        return filename.replace(".tok", ".msk")

    @staticmethod
    def token_filename_to_index_filename(filename):
        assert ".tok" in filename
        return filename.replace(".tok", ".idx")

    @staticmethod
    def merge_padding_and_is_ignore_mask(padding_mask, is_ignore_mask):
        padding_mask = np.bitwise_and(padding_mask, 0x1)
        is_ignore_mask = np.bitwise_and(is_ignore_mask, 0x1)
        return np.bitwise_or(np.left_shift(is_ignore_mask, 1), padding_mask)

    @staticmethod
    def create_by_pad_truncate(token_ids, is_ignore, max_seq_len, padding_value):
        token_ids, padding_mask = pad_truncate_sequences(token_ids, max_seq_len, padding_value, INPUT_IDS_DTYPE)
        if is_ignore:
            is_ignore, _ = pad_truncate_sequences(is_ignore, max_seq_len, 1, np.int8)
        else:
            is_ignore = np.zeros(token_ids.shape, dtype=np.int8)
        index = np.arange(len(token_ids) + 1)

        mask = MmapDataset.merge_padding_and_is_ignore_mask(padding_mask, is_ignore)
        return MmapDataset(token_ids, mask, index)

    def merge(self, rhs):
        if self.mode == DatasetMode.TWO_D:
            token_ids = np.concatenate((self.token_ids, rhs.token_ids))
            mask = np.concatenate((self.mask, rhs.mask))
            index = np.arange(len(token_ids) + 1)
            return MmapDataset(token_ids, mask, index)
        else:
            raise NotImplementedError()


def pad_truncate_sequences(token_ids, max_seq_len, padding_value, dtype):
    cur_seq_len = max(len(x) for x in token_ids)
    padded_token_ids = np.full((len(token_ids), cur_seq_len),
                               TMP_PADDING_VALUE, dtype=dtype)
    for i in range(len(token_ids)):
        padded_token_ids[i, :len(token_ids[i])] = token_ids[i]
    token_ids = padded_token_ids
    if cur_seq_len == max_seq_len:
        pass
    elif cur_seq_len < max_seq_len:
        pad = np.full((token_ids.shape[0], max_seq_len - cur_seq_len), TMP_PADDING_VALUE,
            dtype=dtype)
        token_ids = np.concatenate((token_ids, pad), axis=1)
    else:
        token_ids = np.array(token_ids[:, :max_seq_len])

    padding_mask = (token_ids != TMP_PADDING_VALUE)
    token_ids = np.where(padding_mask, token_ids, padding_value)

    return token_ids, padding_mask.astype(np.int8)
