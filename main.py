import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path + '/kim_cnn')

# The reason of adding kim_cnn to sys.path is:
#    torch.load() could only be executed in the same working directory as model.py
#    This is the way python and pickle works, you need to have the class definition visible/importable when unpickling a file that contains an object.
#    https://github.com/pytorch/pytorch/issues/3678  https://github.com/facebookresearch/InferSent/issues/11

from kim_cnn import KimCNN
model = KimCNN()

input_text_list = ["adfads adsf", "afsddfsa", "a h 89yh hgh l89 h fj lk y7y e45 b i89 kjo 978 kj"]
generated_embedding = model.generate_embedding(input_text_list)
print(generated_embedding)
print(generated_embedding.shape)
