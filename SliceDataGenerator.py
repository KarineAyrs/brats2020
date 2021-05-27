import os
import nibabel as nib
import numpy as np
import cv2


class DataGenerator:
    def __init__(self, pref_paths, batch_size, delimeter, H=None, W=None, h_crop=None, w_crop=None):
        self.__delimeter = delimeter
        self.__batch_size = batch_size
        self.__pref_paths = pref_paths
        self.__brts_20_tr_fldrs = sorted(os.listdir(pref_paths[0]))[:-2]
        self.__brts_16_tr_fldrs = sorted(os.listdir(pref_paths[1]))
        self.__brts_16_tr_lab_fldrs = sorted(os.listdir(pref_paths[2]))
        self.__brts_20_val_fldrs = sorted(os.listdir(pref_paths[3]))[:-2]
        self.__brts_16_val_fldrs = sorted(os.listdir(pref_paths[4]))

        self.__train_X_20_paths, self.__train_Y_20_paths, self.__test_X_20_paths, self.__test_Y_20_paths = self.__create_paths_20()
        self.__train_X_16_paths, self.__train_Y_16_paths, self.__test_X_16_paths, self.__test_Y_16_paths = self.__create_paths_16()

        self.__H = 240 if H is None else H
        self.__W = 240 if W is None else W
        self.__h_crop = 0 if h_crop is None else h_crop
        self.__w_crop = 240 if w_crop is None else w_crop

    def __create_paths_20(self):
        X_paths = []
        Y_paths = []
        for folder in self.__brts_20_tr_fldrs:
            inner_files = os.listdir(self.__pref_paths[0] + self.__delimeter + folder)
            for file in inner_files:
                if file.find('seg') != -1:
                    Y_paths.append(self.__pref_paths[0] + self.__delimeter + folder + self.__delimeter + file)
                else:
                    X_paths.append(self.__pref_paths[0] + self.__delimeter + folder + self.__delimeter + file)
        n = int(len(X_paths) * 0.8)
        n1 = int(len(Y_paths) * 0.8)
        return X_paths[:n], Y_paths[:n1], X_paths[n:], Y_paths[n1:]

    def __create_paths_16(self):
        X_paths = []
        Y_paths = []
        for file in self.__brts_16_tr_fldrs:
            X_paths.append(self.__pref_paths[1] + self.__delimeter + file)
        for file in self.__brts_16_tr_lab_fldrs:
            Y_paths.append(self.__pref_paths[2] + self.__delimeter + file)
        n = int(len(X_paths) * 0.8)
        return X_paths[:n], Y_paths[:n], X_paths[n:], Y_paths[n:]

    def __train_X_slice_sub_data_generator(self):
        nb_tr = len(self.__train_X_16_paths)
        for start in range(0, nb_tr, self.__batch_size):
            x_batch = []
            y_batch = []
            mask = np.array(nib.load(self.__train_Y_16_paths[start]).get_fdata())

            img4d = nib.load(self.__train_X_16_paths[start])
            data = nib.funcs.four_to_three(img4d)

            im1 = np.array(data[0].get_fdata())
            im2 = np.array(data[1].get_fdata())
            im3 = np.array(data[2].get_fdata())
            im4 = np.array(data[3].get_fdata())

            x_batch.clear()
            y_batch.clear()
            i = im1.shape[2] // 2
            mean1 = np.mean(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            mean2 = np.mean(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            mean3 = np.mean(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            mean4 = np.mean(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])

            std1 = np.std(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, im1.shape[2] // 2])
            std2 = np.std(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, im2.shape[2] // 2])
            std3 = np.std(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, im3.shape[2] // 2])
            std4 = np.std(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, im4.shape[2] // 2])

            x_batch.append(
                im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im1[self.__h_crop:self.__w_crop,
                                                                                    self.__h_crop:self.__w_crop,
                                                                                    i].max() == 0 else (im1[
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        i] - mean1) / std1)
            x_batch.append(
                im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im2[self.__h_crop:self.__w_crop,
                                                                                    self.__h_crop:self.__w_crop,
                                                                                    i].max() == 0 else (im2[
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        i] - mean2) / std2)
            x_batch.append(
                im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im3[self.__h_crop:self.__w_crop,
                                                                                    self.__h_crop:self.__w_crop,
                                                                                    i].max() == 0 else (im3[
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        i] - mean3) / std3)
            x_batch.append(
                im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im4[self.__h_crop:self.__w_crop,
                                                                                    self.__h_crop:self.__w_crop,
                                                                                    i].max() == 0 else (im4[
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        self.__h_crop:self.__w_crop,
                                                                                                        i] - mean4) / std4)

            y_batch.append(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i - 5])

            yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
                (1, self.__H, self.__W, 1))

            # for i in range(im1.shape[2]):
            #     x_batch.clear()
            #     y_batch.clear()
            #     mean1 = np.mean(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     mean2 = np.mean(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     mean3 = np.mean(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     mean4 = np.mean(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #     std1 = np.std(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     std2 = np.std(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     std3 = np.std(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #     std4 = np.std(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #     x_batch.append(
            #         im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im1[self.__h_crop:self.__w_crop,
            #                                                                             self.__h_crop:self.__w_crop,
            #                                                                             i].max() == 0 else (im1[
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 i] - mean1) / std1)
            #     x_batch.append(
            #         im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im2[self.__h_crop:self.__w_crop,
            #                                                                             self.__h_crop:self.__w_crop,
            #                                                                             i].max() == 0 else (im2[
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 i] - mean2) / std2)
            #     x_batch.append(
            #         im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im3[self.__h_crop:self.__w_crop,
            #                                                                             self.__h_crop:self.__w_crop,
            #                                                                             i].max() == 0 else (im3[
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 i] - mean3) / std3)
            #     x_batch.append(
            #         im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im4[self.__h_crop:self.__w_crop,
            #                                                                             self.__h_crop:self.__w_crop,
            #                                                                             i].max() == 0 else (im4[
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 self.__h_crop:self.__w_crop,
            #                                                                                                 i] - mean4) / std4)
            #
            #     y_batch.append(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #     yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
            #         (1, self.__H, self.__W, 1))

    def train_X_slice_data_generator(self):
        nb_tr = len(self.__train_X_20_paths)
        while True:
            j = 0
            for start in range(0, nb_tr, self.__batch_size * 4):
                x_batch = []
                y_batch = []
                mask = np.array(nib.load(self.__train_Y_20_paths[j]).get_fdata())
                j += 1

                im1 = np.array(nib.load(self.__train_X_20_paths[start]).get_fdata())
                im2 = np.array(nib.load(self.__train_X_20_paths[start + 1]).get_fdata())
                im3 = np.array(nib.load(self.__train_X_20_paths[start + 2]).get_fdata())
                im4 = np.array(nib.load(self.__train_X_20_paths[start + 3]).get_fdata())

                n = im1.shape[2] // 2

                for i in range(n - 1, n + 2, 1):
                    x_batch.clear()
                    y_batch.clear()

                    x_batch.append(
                        cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    y_batch.append(cv2.resize(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                              (self.__H, self.__W), interpolation=cv2.INTER_AREA))

                    yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
                        (1, self.__H, self.__W, 1))
            # yield from self.__train_X_slice_sub_data_generator()

            #     for i in range(im1.shape[2]):
            #         x_batch.clear()
            #         y_batch.clear()
            #         mean1 = np.mean(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean2 = np.mean(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean3 = np.mean(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean4 = np.mean(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #         std1 = np.std(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std2 = np.std(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std3 = np.std(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std4 = np.std(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #         x_batch.append(
            #             cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean1) / std1)
            #         x_batch.append(
            #             cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean2) / std2)
            #         x_batch.append(
            #             cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean3) / std3)
            #         x_batch.append(
            #             cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean4) / std4)
            #         y_batch.append(cv2.resize(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
            #
            #         yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
            #             (1, self.__H, self.__W, 1))
            # # yield from self.__train_X_slice_sub_data_generator()

    def __test_X_slice_sub_data_generator(self):
        nb_tr = len(self.__test_X_16_paths)
        for start in range(0, nb_tr, self.__batch_size):
            x_batch = []
            y_batch = []
            mask = np.array(nib.load(self.__test_Y_16_paths[start]).get_fdata())

            img4d = nib.load(self.__test_X_16_paths[start])
            data = nib.funcs.four_to_three(img4d)

            im1 = np.array(data[0].get_fdata())
            im2 = np.array(data[1].get_fdata())
            im3 = np.array(data[2].get_fdata())
            im4 = np.array(data[3].get_fdata())

            for i in range(im1.shape[2]):
                x_batch.clear()
                y_batch.clear()
                mean1 = np.mean(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                mean2 = np.mean(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                mean3 = np.mean(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                mean4 = np.mean(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])

                std1 = np.std(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                std2 = np.std(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                std3 = np.std(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
                std4 = np.std(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])

                x_batch.append(
                    im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im1[self.__h_crop:self.__w_crop,
                                                                                        self.__h_crop:self.__w_crop,
                                                                                        i].max() == 0 else (im1[
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            i] - mean1) / std1)
                x_batch.append(
                    im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im2[self.__h_crop:self.__w_crop,
                                                                                        self.__h_crop:self.__w_crop,
                                                                                        i].max() == 0 else (im2[
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            i] - mean2) / std2)
                x_batch.append(
                    im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im3[self.__h_crop:self.__w_crop,
                                                                                        self.__h_crop:self.__w_crop,
                                                                                        i].max() == 0 else (im3[
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            i] - mean3) / std3)
                x_batch.append(
                    im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i] if im4[self.__h_crop:self.__w_crop,
                                                                                        self.__h_crop:self.__w_crop,
                                                                                        i].max() == 0 else (im4[
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            self.__h_crop:self.__w_crop,
                                                                                                            i] - mean4) / std4)
                y_batch.append(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])

                yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
                    (1, self.__H, self.__W, 1))

    def test_X_slice_data_generator(self):
        nb_tr = len(self.__test_X_20_paths)
        while True:
            j = 0
            for start in range(0, nb_tr, self.__batch_size * 4):
                x_batch = []
                y_batch = []
                mask = np.array(nib.load(self.__test_Y_20_paths[j]).get_fdata())
                j += 1
                im1 = np.array(nib.load(self.__test_X_20_paths[start]).get_fdata())
                im2 = np.array(nib.load(self.__test_X_20_paths[start + 1]).get_fdata())
                im3 = np.array(nib.load(self.__test_X_20_paths[start + 2]).get_fdata())
                im4 = np.array(nib.load(self.__test_X_20_paths[start + 3]).get_fdata())

                n = im1.shape[2] // 2

                for i in range(n - 1, n + 2, 1):
                    x_batch.clear()
                    y_batch.clear()

                    x_batch.append(
                        cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    x_batch.append(
                        cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
                    y_batch.append(cv2.resize(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
                                              (self.__H, self.__W), interpolation=cv2.INTER_AREA))

                    yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
                        (1, self.__H, self.__W, 1))
            # yield from self.__test_X_slice_sub_data_generator()
            #     for i in range(im1.shape[2]):
            #         x_batch.clear()
            #         y_batch.clear()
            #         mean1 = np.mean(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean2 = np.mean(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean3 = np.mean(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         mean4 = np.mean(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #         std1 = np.std(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std2 = np.std(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std3 = np.std(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #         std4 = np.std(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i])
            #
            #         x_batch.append(
            #             cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im1[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean1) / std1)
            #         x_batch.append(
            #             cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im2[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean2) / std2)
            #         x_batch.append(
            #             cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im3[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean3) / std3)
            #         x_batch.append(
            #             cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                        (self.__H, self.__W), interpolation=cv2.INTER_AREA)
            #             if im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i].max() == 0
            #             else (cv2.resize(im4[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                              (self.__H, self.__W), interpolation=cv2.INTER_AREA) - mean4) / std4)
            #         y_batch.append(cv2.resize(mask[self.__h_crop:self.__w_crop, self.__h_crop:self.__w_crop, i],
            #                                   (self.__H, self.__W), interpolation=cv2.INTER_AREA))
            #
            #     yield np.array(x_batch).reshape((1, self.__H, self.__W, 4)), np.array(y_batch).reshape(
            #         (1, self.__H, self.__W, 1))
            # # yield from self.__test_X_slice_sub_data_generator()
