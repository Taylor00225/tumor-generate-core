"""
    创建dataset.json
"""
import os
import json
from collections import OrderedDict

path = os.getenv("DemoPath")

path_originalData = path + r"\data"

train_real_image = os.listdir((path_originalData + "\\imagesTr"))
train_real_label = os.listdir((path_originalData + "\\labelsTr"))
test_real_image = os.listdir((path_originalData + "\\imagesTs"))
print(train_real_image)
for idx in range(len(train_real_image)):
    print({'image': "./imagesTr/%s" % train_real_image[idx],
           "label": "./labelsTr/%s" % train_real_label[idx]})

# # -------下面是创建json文件的内容--------------------------
# # 可以根据你的数据集，修改里面的描述
json_dict = OrderedDict()
json_dict['name'] = "PC"  # 任务名
json_dict['description'] = " Segmentation"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "see challenge website"
json_dict['licence'] = "see challenge website"
json_dict['release'] = "0.0"
# 这里填入模态信息，0表示只有一个模态，还可以加入“1”：“MRI”之类的描述，详情请参考官方源码给出的示例
json_dict['modality'] = {"0": "CT"}

# 这里为label文件中的标签，名字可以按需要命名
# # 下面一行在参考的基础上做了修改 # #

json_dict['labels'] = {"Background": 0, "liver": 1}

# 下面部分不需要修改
json_dict['numTraining'] = len(train_real_image)
json_dict['numTest'] = len(test_real_image)
json_dict['file_ending'] = '.nii.gz'

json_dict['training'] = []
for idx in range(len(train_real_image)):
    json_dict['training'].append({'image': "./imagesTr/%s" % train_real_image[idx],
                                  "label": "./labelsTr/%s" % train_real_label[idx]})

json_dict['test'] = ["./imagesTs/%s" % i for i in test_real_image]
with open(os.path.join(path_originalData, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)
