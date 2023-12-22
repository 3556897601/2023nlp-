# 2023nlp-
自然语言处理大作业

采用了pre-train model bert，预训练模型bert-base-chinese的pytorch_model.bin放进对应bert-base-chinese文件夹

第四个epoch的训练的结果
[2023-12-22 10:54:47,630][line: 77] ==> ====Epoch:[4/4] avg_val_loss=0.01198 avg_val_acc=0.99716====
[2023-12-22 10:54:47,630][line: 78] ==> ====Validation epoch took: 07:52====

每一个epoch的微调后的模型会存在cache下的modelX.bin中
