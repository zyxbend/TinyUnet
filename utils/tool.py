# 多个文件中要用到的函数之类的统一写在这里
from skimage.measure import label
import numpy as np
import copy


# 如果最大连通域面积小于2000，直接认为分割错误，返回无分割结果，反之保留面积最大连通域，如果面积第二大连通域和最大差不多，则两个都保留
def refine_output(output):
    refine = np.zeros((1280, 2440), dtype=np.uint8)
    if len(np.where(output > 0)[0]) > 0:
        output = label(output)
        top = output.max()
        area_list = []
        for i in range(1, top + 1):
            area = len(np.where(output == i)[0])
            area_list.append(area)
        max_area = max(area_list)
        max_index = area_list.index(max_area)
        if max_area < 2000:
            return refine
        else:
            refine[output == max_index + 1] = 1
            if top > 1:
                temp_list = copy.deepcopy(area_list)
                del temp_list[max_index]
                second_max_area = max(temp_list)
                second_max_index = area_list.index(second_max_area)
                if (max_area / second_max_area) < 1.2:
                    refine[output == second_max_index + 1] = 1
                    return refine
                else:
                    return refine
            else:
                return refine
    else:
        return refine


# 如果两颗牙的分割结果重合面积大于其中一颗牙的40%，则认为这颗牙分割错误在了其它牙齿上，去掉这颗牙的分割结果
def judge_overlap(id, output_all):
    ids = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43,
           44, 45, 46, 47, 48]
    index = ids.index(id)
    output_id = output_all[:, :, index].reshape(1, -1)  # 每一通道保存着一颗牙的分割结果
    output_id_area = output_id.sum(1) + 0.001
    refine = output_all
    if index <= 29:
        end = index + 3
    elif index == 30:  # 倒数第二颗牙前面只有一颗牙
        end = index + 2
    else:
        end = index + 1  # 最后一颗牙不用再计算重叠率了

    for i in range(index + 1, end):  # 每颗牙和前面两颗牙算重叠区域，因为有可能前面少了一颗牙齿，所以选两颗
        output_other = output_all[:, :, i].reshape(1, -1)
        output_other_area = output_other.sum(1) + 0.001
        inter = (output_id * output_other).sum(1) + 0.001
        if (inter / output_id_area) >= 0.4:
            refine[:, :, index] = 0
        if (inter / output_other_area) >= 0.4:
            refine[:, :, i] = 0
    return refine


# 输入一个模型，获得其参数量
def get_model_params(net):
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print()
