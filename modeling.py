from data_process import read_data,InputDataSet
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch

## 做句子的分类 BertForSequence
class BertForSeq(BertPreTrainedModel):

    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 2 # 类别数目
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
            return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ## loss损失 预测值preds
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )  ## 预测值

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        ## logits -—— softmax层的输入（0.4， 0.6）--- 1
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 二分类任务 这里的参数要做view
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  ##损失
            logits=logits,  ##softmax层的输入，可以理解为是个概率
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
