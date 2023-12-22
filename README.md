# 2023nlp-自然语言处理大作业
采用了pre-train model bert做文本二分类任务

我在自己的笔记本电脑上用GPU训练,显卡是1660ti显存6G,19年的电脑了实在太老土，所以采用的预训练模型进行微调，torch1.13 cuda版本11.6

预训练模型bert-base-chinese的pytorch_model.bin文件太大，可以到[huggingface](https://huggingface.co/bert-base-chinese/blob/main/pytorch_model.bin)下载（地址我都写好了），然后放进对应bert-base-chinese文件夹

每一个epoch的微调后的模型会存在cache下的modelXX.bin中，具体可以看train_and_eval中的训练部分的循环的最后几行，训练的结果模型也太大，提交不上来

训练日志在cache/log下
第四个epoch的训练的结果
[2023-12-22 10:54:47,630][line: 77] ==> ====Epoch:[4/4] avg_val_loss=0.01198 avg_val_acc=0.99716====
[2023-12-22 10:54:47,630][line: 78] ==> ====Validation epoch took: 07:52====
